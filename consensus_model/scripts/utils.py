import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description="Train", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datadir",
                        type=str,
                        help="path to data directory for training",
                        required=True
                        )
    parser.add_argument("--encoder_id",
                        type=int,
                        choices = [0, 1, 2],
                        help=f"0 -> Convolution NN , \n1 -> Transformer encoder-only, \n2 -> MLP",
                        required=True
                        )
    parser.add_argument("-w", "--ways", 
                        type=int,
                        help="number of classes to classify into",
                        required=True
                        )
    parser.add_argument("-s", "--supports",
                        type=int,
                        help="number of examples per class",
                        required=True
                        )
    parser.add_argument("-q", "--queries",
                        type=int,
                        help="number of queries per class",
                        required=True
                        )
    parser.add_argument("--n_hddn",
                        type=int,
                        help="dimension of hidden layers",
                        default=64
                        )
    parser.add_argument("--n_ltnt",
                        type=int,
                        help="dimension in latent space",
                        default=32
                        )
    parser.add_argument("--n_layer",
                        type=int,
                        help="number of layers of transformer encoder"
                        )
    parser.add_argument("--n_head",
                        type=int,
                        help="number of heads in multi-head attention of transformer"
                        )
    parser.add_argument("--dropout",
                        type=float,
                        help="dropout rate (for transformer)"
                        )
    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        help="learing rate",
                        default=0.001
                        )
    parser.add_argument("-ep", "--epoch",
                        type=int,
                        help="number of epoch",
                        default=100
                        )
    
    return parser

