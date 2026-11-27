"""STGCN computation with portable static graph buffers and tensor-only output."""
from copy import deepcopy
import torch
from torch import nn

from libcity.model.traffic_speed_prediction.STGCN import (
    STConvBlock, OutputLayer, calculate_scaled_laplacian, calculate_cheb_poly, calculate_first_approx)


class STGCN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, Ks=3, Kt=3,
                 blocks=None, dropout=0., graph_conv_type='chebconv'):
        super().__init__()
        self.output_window, self.output_dim = output_window, metadata.output_dim
        blocks = deepcopy(blocks or [[1,32,64],[64,32,128]])
        if len(blocks) != 2:
            raise ValueError('STGCN requires two convolution blocks')
        blocks[0][0] = metadata.feature_dim
        remaining = input_window - len(blocks)*2*(Kt-1)
        if remaining <= 0:
            raise ValueError('STGCN input_window is too short for its temporal convolutions')
        graph = metadata.adjacency.copy()
        if graph_conv_type == 'chebconv':
            graph = calculate_cheb_poly(calculate_scaled_laplacian(graph), Ks)
        elif graph_conv_type == 'gcnconv':
            graph, Ks = calculate_first_approx(graph), 1
        else:
            raise ValueError('Unknown STGCN graph convolution')
        self.register_buffer('Lk', torch.as_tensor(graph, dtype=torch.float32))
        self.st_conv1 = STConvBlock(Ks,Kt,metadata.num_nodes,blocks[0],dropout,self.Lk,self.Lk.device)
        self.st_conv2 = STConvBlock(Ks,Kt,metadata.num_nodes,blocks[1],dropout,self.Lk,self.Lk.device)
        for block in (self.st_conv1, self.st_conv2):
            del block.sconv.Lk
            block.sconv.register_buffer('Lk', self.Lk)
        self.output = OutputLayer(blocks[1][2],remaining,metadata.num_nodes,self.output_dim)

    def one_step(self, history):
        value = self.st_conv2(self.st_conv1(history.permute(0,3,1,2)))
        return self.output(value).permute(0,2,3,1)

    def forward(self, batch):
        history = batch['X']
        predictions = []
        for index in range(self.output_window):
            prediction = self.one_step(history)
            predictions.append(prediction)
            following = prediction
            if history.shape[-1] > self.output_dim:
                if 'future_features' not in batch:
                    raise ValueError('STGCN extra channels require future_features')
                following = torch.cat((prediction, batch['future_features'][:,index:index+1]), dim=-1)
            history = torch.cat((history[:,1:], following), dim=1)
        return torch.cat(predictions, dim=1)
