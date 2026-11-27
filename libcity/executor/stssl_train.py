"""Dynamic weighting over epochs, composed with common updates/checkpoints."""
import torch
from libcity.executor.evaluate import evaluate_loop,move_batch
from libcity.executor.train import autocast_context,optimizer_update,complete_epoch


def dynamic_weights(previous,older,temperature):
    ratio=torch.tensor(previous,dtype=torch.float64)/torch.tensor(older,dtype=torch.float64).clamp_min(1e-12)
    return (3*torch.softmax(ratio/temperature,dim=0)).tolist()


def train(context,*,until_epoch=None):
    cfg=context.config['training']
    if set(cfg['strategy'])-{'temperature'} or cfg['strategy'].get('temperature',2.)<=0:
        raise ValueError('STSSL strategy accepts positive temperature only')
    end=cfg['epochs'] if until_epoch is None else min(cfg['epochs'],until_epoch)
    if context.stopper.stopped:
        end=context.state.epoch
    state=context.state.task_state
    older=state.get('older_losses',[1.,1.,1.])
    previous=state.get('previous_losses',[1.,1.,1.])
    for epoch in range(context.state.epoch,end):
        weights=dynamic_weights(previous,older,cfg['strategy'].get('temperature',2.)) if epoch>1 else [1.,1.,1.]
        context.model.train()
        context.optimizer.zero_grad(set_to_none=True)
        weighted_loss=0.
        sample_count=0
        group_weight=0
        separate=torch.zeros(3,dtype=torch.float64)
        for index,batch in enumerate(context.bundle.train):
            batch=move_batch(batch,context.device)
            with autocast_context(context):
                prediction,temporal,spatial=context.model.forward_with_aux({'X':batch['X']})
                prediction_loss=context.loss_fn(prediction,batch['y'],batch.get('y_raw'))
                components=[prediction_loss.loss,temporal,spatial]
                loss=sum(weight*value for weight,value in zip(weights,components))
            if not torch.isfinite(loss):
                raise ValueError('STSSL auxiliary objective must be finite')
            size=len(batch['X'])
            context.amp_scaler.scale(loss*size).backward()
            group_weight+=size
            separate+=torch.tensor([value.detach().item() for value in components])*size
            weighted_loss+=loss.item()*size
            sample_count+=size
            context.state.batches_seen+=1
            if (index+1)%cfg['accumulation_steps']==0 or index+1==len(context.bundle.train):
                optimizer_update(context,group_weight)
                group_weight=0
        older,previous=previous,(separate/sample_count).tolist()
        state.update(older_losses=older,previous_losses=previous,loss_weights=weights,
                     dynamic_weighting_epoch=epoch+1)
        metrics=evaluate_loop(context.model,context.bundle.val,device=context.device,loss_fn=context.loss_fn,
            evaluation=context.config['evaluation'],prefix='val_').metrics
        context.state.epoch=epoch+1
        if complete_epoch(context,metrics,weighted_loss/sample_count):
            break
    if context.checkpoints.best_state is None:
        raise ValueError('No finite STSSL validation result is available')
    context.model.load_state_dict(context.checkpoints.best_state)
    return float(context.state.best_metric)
