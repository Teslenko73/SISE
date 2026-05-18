import sys
import logging

from pprint import pprint as pp
from datetime import datetime
import click

from puzzle.utils import get_cytoband_coord

@click.command()
@click.option('-c','--chrom',
    type = str
)
@click.option('-p','--pos',
    type = int
)
def test_cytoband(chrom, pos):
    """Test to get cytoband coord"""
    if not (chrom and pos):
        print("Please provide coordinates")
        sys.exit()
        
    print(get_cytoband_coord(chrom, pos))

if __name__ == '__main__':
    test_cytoband()
