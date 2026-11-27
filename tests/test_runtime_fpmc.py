"""Personalized Markov math, proper NLL and full ranking lifecycle."""
import json

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.FPMC import FPMC as ReferenceFPMC
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import task_fixture
from runtime_helpers import make_context


def configuration(root, experiment):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='FPMC',params=dict(embedding_size=4)),
        training=dict(epochs=2,batch_size=8,shuffle=True),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


class ReferenceInput(dict):
    def get_origin_len(self, key):
        assert key == 'current_loc'
        return self['lengths'].tolist()


def test_scores_valid_nll_and_gradients_match_markov_factorization(tmp_path):
    task_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference'))
    try:
        old = ReferenceFPMC(dict(embedding_size=4,device=torch.device('cpu')), dict(context.bundle.metadata.info))
        context.model.load_state_dict(old.state_dict())
        batch = next(iter(context.bundle.train))
        expected_scores = old(ReferenceInput(batch))[:, 1:]
        actual = context.model(context.loss_fn.inputs(batch))
        torch.testing.assert_close(actual, F.log_softmax(expected_scores, dim=-1))
        torch.testing.assert_close(actual.exp().sum(dim=-1), torch.ones(len(actual)))
        expected_loss = F.cross_entropy(expected_scores, batch['y'])
        actual_loss = context.loss_fn(actual, batch['y']).loss
        torch.testing.assert_close(actual_loss, expected_loss)
        expected_loss.backward()
        actual_loss.backward()
        for key, value in context.model.named_parameters():
            torch.testing.assert_close(value.grad, dict(old.named_parameters())[key].grad)
        assert set(batch['uid'].tolist()) <= {0, 1}
    finally:
        context.tracker.close()


def test_training_independent_ranking_and_exact_resume(tmp_path):
    task_fixture(tmp_path/'raw')
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics['loss'] == full.metrics['loss']
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state'] == actual['train_state']
    for key,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)
    predictions = json.loads(evaluated.artifacts['predictions'].read_text())
    assert all(len(row['prediction']) == 8 and row['truth'] in range(10,18) for row in predictions)


def test_batch_one_padding_label_isolation_and_float64(tmp_path):
    task_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'padding'))
    try:
        model = context.model.eval().double()
        batch = {key:value[:1] for key,value in next(iter(context.bundle.test)).items()}
        prediction = model(context.loss_fn.inputs(batch))
        extended = {**batch,'current_loc':F.pad(batch['current_loc'],(0,3),value=9999),
                    'y':torch.full_like(batch['y'],9999)}
        torch.testing.assert_close(prediction,model(extended),rtol=0,atol=0)
        assert prediction.shape == (1,8) and prediction.dtype == torch.float64
        assert all(not any(hasattr(module,key) for key in ('device','_logger','_scaler')) for module in model.modules())
    finally:
        context.tracker.close()


def test_personalized_user_vocabulary_is_validated(tmp_path):
    folder = task_fixture(tmp_path/'raw')
    pd.DataFrame({'usr_id':[100]}).to_csv(folder/'TaskTiny.usr',index=False)
    with pytest.raises(ValueError,match='user is absent'):
        run_experiment('train',config=configuration(tmp_path,'invalid-user'))
