import sys
import logging
import subprocess

from pprint import pprint as pp
from datetime import datetime
import click

@click.command()
@click.argument('vcf',
    nargs=1,
    type=click.Path(exists=True),
    # type=click.Path(exists=True),
)
@click.option('-v', '--variant',
    nargs=1,
    type=int
)
def test_grep(vcf, variant):
    """Test grep"""
    if vcf.endswith('.gz'):
        cat = subprocess.Popen(['zcat', vcf], 
                        stdout=subprocess.PIPE,
                        )
    else:
        cat = subprocess.Popen(['cat', vcf], 
                                stdout=subprocess.PIPE,
                                )

    grep_commands = ["grep", "BND\|DEL"]
    grep_commands = ["grep", "BND"]
    
    grep = subprocess.Popen(grep_commands,
                        stdin=cat.stdout,
                        stdout=subprocess.PIPE,
                        )
    
    end_of_pipe = grep.stdout
    
    for line in end_of_pipe:
        if not line.startswith('#'):
            print(line.split('\t'))

    


if __name__ == '__main__':
    test_grep()
