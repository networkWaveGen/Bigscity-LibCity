"""Independent multi-update trainer composed from common runtime components."""
import numpy as np
import torch
from libcity.executor.evaluate import evaluate_loop,move_batch
from libcity.executor.train import autocast_context,optimizer_update,complete_epoch,backward


def updates_per_epoch(config,bundle):
    strategy = config['training']['strategy']
    allowed = {'num_split','permutation_interval','curriculum_step','use_curriculum_learning'}
    if strategy.keys()-allowed:
        raise ValueError('Unknown MTGNN training.strategy keys')
    splits = strategy['num_split']
    if type(splits) is not int or not 1<=splits<=bundle.metadata.num_nodes:
        raise ValueError('MTGNN num_split must be between 1 and num_nodes')
    for key in ('permutation_interval','curriculum_step'):
        if type(strategy[key]) is not int or strategy[key]<1:
            raise ValueError(f'MTGNN {key} must be a positive integer')
    if type(strategy['use_curriculum_learning']) is not bool:
        raise TypeError('MTGNN use_curriculum_learning must be boolean')
    if config['training']['accumulation_steps']!=1:
        raise ValueError('MTGNN independent subgraph updates require accumulation_steps=1')
    return len(bundle.train)*splits


def subgraphs(permutation,splits):
    width = len(permutation)//splits
    return [permutation[index*width:(index+1)*width] if index<splits-1 else permutation[index*width:]
            for index in range(splits)]


def train(context, *,until_epoch=None):
    updates_per_epoch(context.config,context.bundle)
    strategy = context.config['training']['strategy']
    context.state.stage = 'subgraphs'
    end = context.config['training']['epochs']
    if until_epoch is not None:
        end = min(end,until_epoch)
    if context.stopper.stopped:
        end = context.state.epoch
    for epoch in range(context.state.epoch,end):
        optimizer_steps_before = context.state.optimizer_steps
        context.model.train()
        total,weight = 0.,0
        for index,batch in enumerate(context.bundle.train):
            batch = move_batch(batch,context.device)
            if index%strategy['permutation_interval']==0:
                permutation = np.random.permutation(context.bundle.metadata.num_nodes)
                context.state.task_state['permutation'] = permutation.tolist()
            for nodes in subgraphs(permutation,strategy['num_split']):
                nodes = torch.as_tensor(nodes,dtype=torch.long,device=context.device)
                context.optimizer.zero_grad(set_to_none=True)
                progress = context.state.task_state.get('subgraph_steps',0)
                level = min(context.config['data']['output_window'],1+progress//strategy['curriculum_step'])
                with autocast_context(context):
                    prediction = context.model({'X':batch['X'][:,:,nodes], 'node_idx':nodes})
                    target = batch['y'][:,:,nodes]
                    raw_target = batch['y_raw'][:,:,nodes] if 'y_raw' in batch else None
                    if strategy['use_curriculum_learning']:
                        prediction,target = prediction[:,:level],target[:,:level]
                        raw_target = raw_target[:,:level] if raw_target is not None else None
                    result = context.loss_fn(prediction,target,raw_target)
                if not torch.isfinite(result.loss):
                    raise ValueError('MTGNN training loss must be finite')
                backward(context,result.loss*result.weight)
                optimizer_update(context,result.weight)
                context.state.task_state['subgraph_steps'] = progress+1
                context.state.task_state['task_level'] = level
                total += result.loss.item()*result.weight
                weight += result.weight
            context.state.batches_seen += 1
        metrics = evaluate_loop(context.model,context.bundle.val,device=context.device,loss_fn=context.loss_fn,
                                evaluation=context.config['evaluation'],prefix='val_').metrics
        context.state.epoch = epoch+1
        if complete_epoch(context,metrics,total/weight if weight else 0.,optimizer_steps_before=optimizer_steps_before):
            break
    if context.checkpoints.best_state is None:
        raise ValueError('No completed MTGNN validation state available')
    context.model.load_state_dict(context.checkpoints.best_state)
    return float(context.state.best_metric)
