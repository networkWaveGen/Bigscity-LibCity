"""Gensim fitting with streamed walks, safe weights and explicit diagnostics."""
import json
import zlib

import numpy as np
import torch

from libcity.data.walk_graph import WalkCorpus
from libcity.executor.evaluate import ExperimentResult,write_metrics
from libcity.utils.runtime_checkpoint import CheckpointManager
from libcity.data.sources import atomic_sources


def stable_token_hash(value):
    return zlib.crc32(str(value).encode('utf-8'))


class WalkLines:
    def __init__(self,path):
        self.path = path

    def __iter__(self):
        with self.path.open(encoding='utf-8') as stream:
            for line in stream: yield line.split()


def train(context):
    try:
        import gensim
        from gensim.models import Word2Vec
    except (ImportError,ValueError) as error:
        raise ImportError('Walk training requires NumPy-2-compatible Gensim: pip install -e ".[embeddings]" (gensim>=4.4,<5)') from error
    model,training = context.model,context.config['training']
    corpus = WalkCorpus(context.bundle.metadata,algorithm=model.algorithm,num_walks=model.num_walks,
        walk_length=model.walk_length,p=model.p,q=model.q,restart=model.restart,seed=context.config['runtime']['seed'])
    folder = context.directory/'training';folder.mkdir(parents=True,exist_ok=True)
    path = folder/'walks.txt'
    walks,words = 0,0
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        for walk in corpus:
            stream.write(' '.join(walk)+'\n');walks += 1;words += len(walk)
    options = dict(vector_size=model.vectors.shape[1],window=training['window_size'],min_count=0,sg=1,
        hs=int(training['loss']['hierarchical_softmax']),negative=training['loss']['negative_samples'],
        workers=training['num_workers'],epochs=training['epochs'],alpha=training['optimizer']['lr'],
        min_alpha=training['optimizer']['min_lr'],sample=training['subsample'],shrink_windows=training['dynamic_window'],
        batch_words=training['batch_words'],seed=context.config['runtime']['seed'],hashfxn=stable_token_hash)
    class Progress:
        def __init__(self): self.epoch,self.loss = 0,0.
        def on_train_begin(self,backend): pass
        def on_train_end(self,backend): pass
        def on_epoch_begin(self,backend): pass
        def on_epoch_end(self,backend):
            self.epoch += 1
            total = float(backend.get_latest_training_loss())
            record = dict(stage='word2vec',epoch=self.epoch,training_loss=total-self.loss)
            self.loss = total
            with (context.directory/'metrics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            context.logger.info('%s',record)
    backend = Word2Vec(**options)
    sentences = WalkLines(path)
    backend.build_vocab(sentences)
    trained,raw = backend.train(sentences,total_examples=backend.corpus_count,epochs=training['epochs'],compute_loss=True,callbacks=[Progress()])
    vectors = np.stack([backend.wv[str(index)] for index in range(len(model.vectors))])
    if not np.isfinite(vectors).all(): raise ValueError('Word2Vec produced nonfinite node vectors')
    model.vectors.copy_(torch.from_numpy(vectors));model.fitted.fill_(True)
    checkpoint = CheckpointManager(context.directory/'checkpoints').save_weights(model,'embeddings.pt')
    report = dict(backend='gensim',version=gensim.__version__,algorithm=model.algorithm,epochs=training['epochs'],
        walks=walks,walk_tokens=words,trained_words=int(trained),raw_training_words=int(raw),
        word2vec_loss=float(backend.get_latest_training_loss()),workers=training['num_workers'],
        seed=context.config['runtime']['seed'],hierarchical_softmax=bool(options['hs']),negative_samples=options['negative'],
        validation=False,resume=False,isolated_nodes=context.bundle.metadata.info['isolated_nodes'],
        atomic_sources=atomic_sources(context.config['data']))
    report_path = folder/'report.json';report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return ExperimentResult({},dict(weights=checkpoint,training_report=report_path,walks=path,training_metrics=context.directory/'metrics.jsonl'))


def evaluate(model,bundle,evaluation,*,directory,logger):
    model.eval()
    values = model.embeddings().detach().cpu().numpy()
    diagnostics = dict(nodes=float(len(values)),dimensions=float(values.shape[1]),coverage=1.,
        mean_norm=float(np.linalg.norm(values.astype(np.float64),axis=1).mean()),isolated_nodes=float(bundle.metadata.info['isolated_nodes']),
        dead_end_nodes=float(bundle.metadata.info['dead_end_nodes']))
    metrics = {name:diagnostics[name] for name in evaluation['metrics']}
    artifacts = write_metrics(metrics,directory)
    vectors,identities = directory/'embeddings.npy',directory/'node_ids.json'
    if evaluation['save_predictions']:
        np.save(vectors,values,allow_pickle=False)
        identities.write_text(json.dumps(list(bundle.metadata.info['node_ids']),ensure_ascii=False)+'\n',encoding='utf-8')
        artifacts.update(embeddings=vectors,node_ids=identities)
    else:
        vectors.unlink(missing_ok=True);identities.unlink(missing_ok=True)
    return ExperimentResult(metrics,artifacts)
