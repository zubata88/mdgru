from options.parser import get_parser


def get_args(args=None):
    return get_parser().parse_args(args=args)

