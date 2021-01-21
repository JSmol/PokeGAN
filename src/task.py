import argparse
def get_args():
  args_parser = argparse.ArgumentParser()
  # Train arguments
  args_parser.add_argument(
    '--epochs',
    default=1,
    type=int,
  )
  args_parser.add_argument(
    '--batch-size',
    default=89,
    type=int,
  )
  args_parser.add_argument(
    '--learning-rate',
    default=0.002,
    type=float,
  )
  args_parser.add_argument(
    '--weight-decay',
    default=0,
    type=float,
  )
  args_parser.add_argument(
    '--beta1',
    default=0.5,
    type=float,
  )
  args_parser.add_argument(
    '--beta2',
    default=0.999,
    type=float,
  )

  # model args
  args_parser.add_argument(
    '--load-model',
    default=False,
    type=bool,
  )
  
  return args_parser.parse_args()

from model import get_model, save_model
from data import load_data
from train import train

if __name__ == '__main__':
  args = get_args()
  print('task running with args:')
  print(vars(args))

  G, D = get_model(args.load_model)
  print('-- model loaded --')

  dataset = load_data('.')
  print('-- data loaded --')

  train(G, D, dataset,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.learning_rate,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay
  )

  save_model(G, 'FINAL', 'G')
  save_model(D, 'FINAL', 'D')
  print('-- models uploaded --')
  print('-- task complete --')
