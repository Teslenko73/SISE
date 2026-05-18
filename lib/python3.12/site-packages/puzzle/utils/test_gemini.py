import sys
import logging

from pprint import pprint as pp
from datetime import datetime
import click

from gemini import GeminiQuery
from gemini.gim import (AutoDom, AutoRec, DeNovo, MendelViolations, CompoundHet)

class Args(object):
    """
    >>> args = Args(db='some.db')
    >>> args.db
    'some.db'
    """
    _opts = ("columns", "db", "filter", "min_kindreds", "families",
                 "pattern_only", "max_priority", # only for compound_het
                 "allow_unaffected", "min_gq", "lenient", "min_sample_depth")
    def __init__(self, **kwargs):
        if not "min_gq" in kwargs: kwargs['min_gq'] = 0
        if not "lenient" in kwargs: kwargs['lenient'] = False
        for k in ("families", "filter"):
            if not k in kwargs: kwargs[k] = None
        if not "gt_phred_ll" in kwargs: kwargs['gt_phred_ll'] = None
        if not "min_sample_depth" in kwargs: kwargs['min_sample_depth'] = 0

        for k in ("min_kindreds", "max_priority"):
            if not k in kwargs: kwargs[k] = 1
        for k in ("pattern_only", "allow_unaffected"):
            if not k in kwargs: kwargs[k] = False
        self.__dict__.update(**kwargs)


def check_genotypes(gemini_variant, index):
    """Check the genotypes for a variant"""
    print(gemini_variant['gts'][index])
    print(gemini_variant['gt_types'][index])

@click.command()
@click.argument('db',
    type=click.Path(exists=True),
    # type=click.Path(exists=True),
)
@click.option('-v', '--variant',
    type=int
)
@click.option('--value',
)
@click.option('--query',
    default="SELECT * from variants"
)
def test_gemini(db, variant, value, query):
    """Test a gemini db"""
    gq = GeminiQuery(db)
    
    query = "SELECT * from variants"
    # query = "SELECT * from samples"
    
    # query = "gemini comp_hets"
    
    # if variant:
    #     query = "SELECT * from variants WHERE (aaf_1kg_all < {0} or aaf_1kg_all is Null)".format(variant)
    # else:
    #      query = "select * from variants"
    
    # query = "SELECT * from variants WHERE impact_severity in ('HIGH', 'LOW')"
    gq.run(query)
    # sample2idx = gq.sample_to_idx
    nr_of_variants = 0
    for index, variant in enumerate(gq):
        nr_of_variants += 1
    print(nr_of_variants)
    sys.exit()
    #     for key in variant.keys():
    #         if value:
    #             if key == value:
    #                 print("{0}: {1}".format(key, variant[key]))
    #         else:
    #             print("'{0}': {1},".format(key, variant[key]))
    #     print("")
        # check_genotypes(variant, 0)
        # check_genotypes(variant, 1)
        # check_genotypes(variant, 2)
    
    # print(sample2idx)
    # print(index)
    models = []
    #chrom,start,end,impact_severity,impact,gene,transcript,max_aaf_all
    models.append(AutoDom(Args(db=db,
                         columns="*")))
    models.append(AutoRec(Args(db=db,
                         columns="*")))
    models.append(DeNovo(Args(db=db,
                         columns="*",
                         families='643594')))
    models.append(CompoundHet(Args(db=db,
                         columns="*",
                         families='643594')))
    
    for model in models:
        variants = model.report_candidates()
        print(variants)
        print(model)
        nr_of_variants = 0
        for i, row in enumerate(variants):
            nr_of_variants += 1
            # print(i,row)
            # row is an OrderedDict
            # for key in row.keys():
            #     print("{0}:\t{1}".format(key, row[key]))
            # if i == 0:
            #     print "\t".join(row.keys())
            # print "\t".join(map(str, row.values()))
        print(nr_of_variants)
        print("")


if __name__ == '__main__':
    test_gemini()
