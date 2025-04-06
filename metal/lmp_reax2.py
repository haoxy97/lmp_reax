#!/usr/bin/env python3
from reax2.case import Case,remove_cache
from utility import ArgvParser


def run(path=None):
    argv = ArgvParser(default_path='.')
    argv.parser.add_argument('-sf', '--skipFrame', help='[10]', type=int, default=10)
    argv.parser.add_argument('-c', '--cache', help='use cache, [True]', default=True, action='store_false')
    argv.parser.add_argument('-r', '--remove', help='remove uncomplete cache[False]', default=False, action='store_true')
    if argv.remove:
        remove_cache(argv.path)
        exit()
    if path is not None:
        argv.path = path
    if argv.debug:
        argv.path = '/Users/hao/code/reax2/example'
        case = Case(argv.path, skip_frame=argv.skipFrame, use_cache=argv.cache)
        # case.dump_specie()
        # case.plot_rxn_path()
    else:
        case = Case(argv.path, skip_frame=argv.skipFrame, use_cache=argv.cache)
        case.plot()
        case.dump_all()


if __name__ == '__main__':
    run()
