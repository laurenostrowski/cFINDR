## Initial architecture search (gating disabled)
# python main.py --datapath=data.npz --workdir=checkpoints

## Lambda sparse search with previously defined hyperparams (gating enabled)
# python main.py --datapath=data.npz --workdir=checkpoints \
#   --learning_rate=0.01 --feature_size=30 --net_size=100 --latent_size=10 --search_sparse

## Single run with specified params (no search)
# python main.py --datapath=data.npz --workdir=checkpoints \
#   --learning_rate=0.01 --feature_size=30 --net_size=100 --latent_size=10

from absl import app
from absl import flags
from absl import logging
import ml_collections
import numpy as np
import optuna
import os
import train

FLAGS = flags.FLAGS

flags.DEFINE_string('datapath', None, 'Path to data.')
flags.DEFINE_string('workdir', None, 'Directory to store model fits.')
flags.DEFINE_float('learning_rate', None, 'Learning rate (None = run architecture search)')
flags.DEFINE_integer('feature_size', None, 'Feature size')
flags.DEFINE_integer('net_size', None, 'Network size')
flags.DEFINE_integer('latent_size', None, 'Latent size')
flags.DEFINE_float('lambda_sparse', 1e-4, 'Lambda sparse')
flags.DEFINE_boolean('search_sparse', False, 'Run Optuna search for lambda_sparse')
flags.DEFINE_integer('seed', 0, 'JAX random seed for model initialization')
flags.DEFINE_integer('k_cv', 1, 'Cross-validation fold to train (1-5)')
flags.DEFINE_integer('num_epochs', 3000, 'Number of training epochs')
flags.DEFINE_string('basedir', None, 'Base directory for baseline.npy (defaults to parent of workdir)')
flags.DEFINE_float('alpha', 0.039, 'Euler step in units of the time constant: bin width / tau '
                   '(0.039 = 7.8 ms / 200 ms at 128 Hz; 0.2 = 40 ms / 200 ms)')
flags.DEFINE_integer('anneal_epochs', 200, 'Epochs for the KL weight to reach 99 %; best-validation selection '
                     'starts after them (original: 458, beta_inc_rate = 0.99)')
flags.DEFINE_float('beta', 2.0, 'Weight of the KL term between posterior and prior drifts (original: 2)')
flags.DEFINE_boolean('posterior_language', True, 'Give the encoder / posterior the language one-hot (inputs 0, 1). '
                     'False: withheld (posterior must infer language from the data); the prior flow always gets it')
flags.DEFINE_float('l2_prior', 1e-3, 'L2 on prior-flow weight matrices')
flags.DEFINE_float('l2_posterior', 1e-3, 'L2 on posterior-SDE weight matrices')
flags.DEFINE_float('l2_encoder', 1e-4, 'L2 on inference-GRU weight matrices')
flags.DEFINE_float('l2_readout', 1e-4, 'L2 on readout weight matrix')
flags.DEFINE_boolean('balance_languages', False, 'Train on a random language-balanced subset of trials drawn from '
                     '--seed (the larger language subsampled to the smaller one\'s count; the folds split that subset)')

def get_config(k_cv, num_epochs, learning_rate, feature_size, net_size, latent_size, lambda_sparse=1e-4, seed=0):
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.k_cv = k_cv
  config.seed = seed
  config.base_learning_rate = learning_rate
  config.features_prior = [feature_size]
  config.features_posterior = [feature_size]
  config.inference_network_size = net_size
  config.beta = FLAGS.beta  # KL weight: higher = posterior latents must follow the prior flow more closely
  config.noise_level = 1.
  config.cosine_epochs = 190
  config.alpha = FLAGS.alpha  # bin width / tau (tau = 200 ms): 0.039 at 128 Hz, 0.2 at 40 ms bins
  config.task_related_latent_size = latent_size
  config.n_splits = 5 # 5-fold cross-validation
  config.l2_prior = FLAGS.l2_prior          # L2 on weight matrices per module (replaces l2_coeff, which was
  config.l2_posterior = FLAGS.l2_posterior  # applied only to a module this model does not have)
  config.l2_encoder = FLAGS.l2_encoder
  config.l2_readout = FLAGS.l2_readout
  config.lambda_sparse = lambda_sparse # L1 sparsity on channel gates
  config.cosine_mult_by = 2
  config.warmup_epochs = 10
  config.batch_size = 5
  config.num_epochs = num_epochs
  config.momentum = 0.9
  config.beta_inc_rate = float(0.01 ** (1. / FLAGS.anneal_epochs))  # KL weight 1 - rate**epoch reaches 99 % at anneal_epochs
  config.lossw_inc_rate = 1.  # decreasing this value increases lossw faster
  config.earlymiddle_epochs = 0
  config.baseline_fit = False
  config.constrain_prior = False
  config.use_gaussian = True        # added for sEEG data
  config.use_channel_gating = True  # added for sEEG data
  config.posterior_input_mask = None if FLAGS.posterior_language else [0., 0., 1., 1.]  # [dutch, english, image, speaking]
  config.balance_languages = FLAGS.balance_languages

  return config

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # basedir: where baseline.npy lives. defaults to parent of workdir.
  basedir = FLAGS.basedir if FLAGS.basedir is not None else os.path.dirname(FLAGS.workdir)

  # determine mode: architecture search vs lambda search
  run_architecture_search = (FLAGS.learning_rate is None)

  if run_architecture_search:
    # initial architecture search (channel gating disabled)
    def objective(trial):
      # learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.003, 0.01, 0.03, 0.1])
      # feature_size = trial.suggest_categorical('feature_size', [30, 50, 100, 150, 200])
      # net_size = trial.suggest_categorical('net_size', [200, 300, 400, 500]) # ~480 inputs
      # latent_size = trial.suggest_categorical('latent_size', [2, 4, 8, 10, 12, 16])

      ##### sEEG patients
      learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01, log=True)
      feature_size = trial.suggest_int('feature_size', 30, 200, step=25)
      net_size = trial.suggest_int('net_size', 50, 400, step=50)
      latent_size = trial.suggest_int('latent_size', 12, 32, step=4)

      # ##### grid patient -- bipolar montage
      # learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01, log=True)
      # feature_size = trial.suggest_int('feature_size', 10, 50, step=10)
      # net_size = trial.suggest_int('net_size', 20, 160, step=20)
      # latent_size = trial.suggest_int('latent_size', 6, 20, step=2)

      ##### grid patient -- common reference
      # learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01, log=True)
      # feature_size = trial.suggest_int('feature_size', 10, 40, step=5)
      # net_size = trial.suggest_int('net_size', 20, 120, step=20)
      # latent_size = trial.suggest_int('latent_size', 4, 16, step=2)
      
      config = get_config(
        k_cv=1, 
        num_epochs=1500, 
        learning_rate=learning_rate,
        feature_size=feature_size,
        net_size=net_size,
        latent_size=latent_size,
        lambda_sparse=1e-4
      )
      config.use_channel_gating = False  # disable during architecture search
      return train.train_and_evaluate(config, FLAGS.datapath, FLAGS.workdir, ckpt_save=False,
                                      basedir=basedir)

    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    
    # final run with best architecture params
    config = get_config(
      k_cv=1, 
      num_epochs=3000,
      learning_rate=study.best_params['learning_rate'],
      feature_size=study.best_params['feature_size'],
      net_size=study.best_params['net_size'],
      latent_size=study.best_params['latent_size'],
      lambda_sparse=1e-4
    )
    config.use_channel_gating = False
    train.train_and_evaluate(config, FLAGS.datapath, FLAGS.workdir, ckpt_save=True,
                             basedir=basedir)
    logging.info('Best architecture params: %s', study.best_params)
    
  elif FLAGS.search_sparse:
    # lambda sparse search with fixed architecture
    def objective(trial):
      lambda_sparse = trial.suggest_float('lambda_sparse', 1e-4, 0.1, log=True)
      
      config = get_config(
        k_cv=1, 
        num_epochs=1500, 
        learning_rate=FLAGS.learning_rate,
        feature_size=FLAGS.feature_size,
        net_size=FLAGS.net_size,
        latent_size=FLAGS.latent_size,
        lambda_sparse=lambda_sparse
      )
      config.use_channel_gating = True  # enabled for lambda search
      return train.train_and_evaluate(config, FLAGS.datapath, FLAGS.workdir, ckpt_save=False,
                                      basedir=basedir)

    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    
    # final run with best lambda
    config = get_config(
      k_cv=1, 
      num_epochs=3000,
      learning_rate=FLAGS.learning_rate,
      feature_size=FLAGS.feature_size,
      net_size=FLAGS.net_size,
      latent_size=FLAGS.latent_size,
      lambda_sparse=study.best_params['lambda_sparse']
    )
    train.train_and_evaluate(config, FLAGS.datapath, FLAGS.workdir, ckpt_save=True,
                             basedir=basedir)
    logging.info('Best lambda_sparse: %s', study.best_params['lambda_sparse'])
    
  else:
    # single run with specified params (no search)
    config = get_config(
      k_cv=FLAGS.k_cv,
      num_epochs=FLAGS.num_epochs,
      learning_rate=FLAGS.learning_rate,
      feature_size=FLAGS.feature_size,
      net_size=FLAGS.net_size,
      latent_size=FLAGS.latent_size,
      lambda_sparse=FLAGS.lambda_sparse,
      seed=FLAGS.seed
    )
    train.train_and_evaluate(config, FLAGS.datapath, FLAGS.workdir, ckpt_save=True, basedir=basedir)

if __name__ == '__main__':
  flags.mark_flags_as_required(['datapath', 'workdir'])
  app.run(main)