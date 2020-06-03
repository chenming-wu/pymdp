#!
# author: Jun Xu and Tie-Yan Liu
# modified by Jun Xu, June 12, 2010 (for Microsoft Learning to Rank Datasets)
use strict;

#hash table for NDCG,
my %hsNdcgRelScore = (  "4", 15,
						"3", 7,
						"2", 3,
                        "1", 1,
                        "0", 0,
                    );

#hash table for Precision@N and MAP
my %hsPrecisionRel = ("4", 1,
                      "3", 1,
					  "2", 1,
					  "1", 0,
                      "0", 0,
                    );
my $iMaxPosition = 10;

my $argc = $#ARGV+1;
if($argc != 4)
{
		print "Invalid command line.\n";
		print "Usage: perl $0 argv[1] argv[2] argv[3] argv[4]\n";
		print "argv[1]: feature file \n";
		print "argv[2]: prediction file\n";
		print "argv[3]: result (output) file\n";
		print "argv[4]: flag. If flag equals 1, output the evaluation results per query; if flag equals 0, simply output the average results.\n";
		exit -1;
}
my $fnFeature = $ARGV[0];
my $fnPrediction = $ARGV[1];
my $fnResult = $ARGV[2];
my $flag = $ARGV[3];
if($flag != 1 && $flag != 0)
{
	print "Invalid command line.\n";
	print "Usage: perl Eval.pl argv[1] argv[2] argv[3] argv[4]\n";
	print "Flag should be 0 or 1\n";
	exit -1;
}

my %hsQueryDocLabelScore = ReadInputFiles($fnFeature, $fnPrediction);
my %hsQueryEval = EvalQuery(\%hsQueryDocLabelScore);
OuputResults($fnResult, %hsQueryEval);


sub OuputResults
{
    my ($fnOut, %hsResult) = @_;
    open(FOUT, ">$fnOut");

    my @qids = sort{$a <=> $b} keys(%hsResult);
    my $numQuery = @qids;
    
# #Precision@N and MAP
# # modified by Jun Xu, March 3, 2009
# # changing the output format
#     print FOUT "qid\tP\@1\tP\@2\tP\@3\tP\@4\tP\@5\tP\@6\tP\@7\tP\@8\tP\@9\tP\@10\tMAP\n";
# #---------------------------------------------
#     my @prec;
#     my $map = 0;
#     for(my $i = 0; $i < $#qids + 1; $i ++)
#     {
# # modified by Jun Xu, March 3, 2009
# # output the real query id
#         my $qid = $qids[$i];
#         my @pN = @{$hsResult{$qid}{"PatN"}};
#         my $map_q = $hsResult{$qid}{"MAP"};
#         if ($flag == 1)
#         {
#             print FOUT "$qid\t";
#             for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
#             {
#                 print FOUT sprintf("%.4f\t", $pN[$iPos]);
#             }
#             print FOUT sprintf("%.4f\n", $map_q);
#         }
#         for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
#         {
#             $prec[$iPos] += $pN[$iPos];
#         }
#         $map += $map_q;
#     }
#     print FOUT "Average\t";
#     for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
#     {
#         $prec[$iPos] /= ($#qids + 1);
#         print FOUT sprintf("%.4f\t", $prec[$iPos]);
#     }
#     $map /= ($#qids + 1);
#     print FOUT sprintf("%.4f\n\n", $map);
    
#NDCG and MeanNDCG
# modified by Jun Xu, March 3, 2009
# changing the output format
    # print FOUT "qid\tNDCG\@1\tNDCG\@2\tNDCG\@3\tNDCG\@4\tNDCG\@5\tNDCG\@6\tNDCG\@7\tNDCG\@8\tNDCG\@9\tNDCG\@10\tMeanNDCG\n";
#---------------------------------------------
    my @ndcg;
    # my $meanNdcg = 0;
    for(my $i = 0; $i < $#qids + 1; $i ++)
    {
# modified by Jun Xu, March 3, 2009
# output the real query id
        my $qid = $qids[$i];
        my @ndcg_q = @{$hsResult{$qid}{"NDCG"}};
        # my $meanNdcg_q = $hsResult{$qid}{"MeanNDCG"};
        if ($flag == 1)
        {
            print FOUT "$qid\t";
            for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
            {   if ($iPos == 0 or $iPos == 2 or $iPos == 4 or $iPos == 9)
                {
                    print FOUT sprintf("%.4f\t", $ndcg_q[$iPos]);
                }
            }
            # print FOUT sprintf("%.4f\n", $meanNdcg_q);
        }
        for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
        {
            $ndcg[$iPos] += $ndcg_q[$iPos];
        }
        # $meanNdcg += $meanNdcg_q;
    }
    print FOUT "- Eval metrics:";
    for(my $iPos = 0; $iPos < $iMaxPosition; $iPos ++)
    {
        $ndcg[$iPos] /= ($#qids + 1);
        if ($iPos == 0 or $iPos == 2 or $iPos == 4)
        {
            print FOUT sprintf(" ndcg_%d: %.4f ;", $iPos+1, $ndcg[$iPos]);
        }
        if ($iPos == 9)
        {
            print FOUT sprintf(" ndcg_%d: %.4f", $iPos+1, $ndcg[$iPos]);
        }

    }
    # $meanNdcg /= ($#qids + 1);
    # print FOUT sprintf("%.4f\n\n", $meanNdcg);

    close(FOUT);
}

sub EvalQuery
{
    my $pHash = $_[0];
    my %hsResults;
    
    my @qids = sort{$a <=> $b} keys(%$pHash);
    for(my $i = 0; $i < @qids; $i ++)
    {
        my $qid = $qids[$i];
        my @tmpDid = sort{$$pHash{$qid}{$a}{"lineNum"} <=> $$pHash{$qid}{$b}{"lineNum"}} keys(%{$$pHash{$qid}});
        my @docids = sort{$$pHash{$qid}{$b}{"pred"} <=> $$pHash{$qid}{$a}{"pred"}} @tmpDid;
        my @rates;

        for(my $iPos = 0; $iPos < $#docids + 1; $iPos ++)
        {
            $rates[$iPos] = $$pHash{$qid}{$docids[$iPos]}{"label"};
        }

        my $map  = MAP(@rates);
        my @PAtN = PrecisionAtN($iMaxPosition, @rates);
# modified by Jun Xu, calculate all possible positions' NDCG for MeanNDCG
        my @Ndcg = NDCG($iMaxPosition, @rates);# modified
        
        # my @Ndcg = NDCG($#rates + 1, @rates);# modified
        my $meanNdcg = 0;
        for(my $iPos = 0; $iPos < $#Ndcg + 1; $iPos ++)
        {
            $meanNdcg += $Ndcg[$iPos];
        }
        $meanNdcg /= ($#Ndcg + 1);
        
        
        @{$hsResults{$qid}{"PatN"}} = @PAtN;
        $hsResults{$qid}{"MAP"} = $map;
        @{$hsResults{$qid}{"NDCG"}} = @Ndcg;
        $hsResults{$qid}{"MeanNDCG"} = $meanNdcg;

    }
    return %hsResults;
}

sub ReadInputFiles
{
    my ($fnFeature, $fnPred) = @_;
    my %hsQueryDocLabelScore;
    
    if(!open(FIN_Feature, $fnFeature))
	{
		print "Invalid command line.\n";
		print "Open \$fnFeature\" failed.\n";
		exit -2;
	}
	if(!open(FIN_Pred, $fnPred))
	{
		print "Invalid command line.\n";
		print "Open \"$fnPred\" failed.\n";
		exit -2;
	}

    my $lineNum = 0;
    while(defined(my $lnFea = <FIN_Feature>))
    {
        $lineNum ++;
        chomp($lnFea);
        my $predScore = <FIN_Pred>;
        if (!defined($predScore))
        {
            print "Error to read $fnPred at line $lineNum.\n";
            exit -2;
        }
        chomp($predScore);
        if ($lnFea =~ m/^(\d+) qid\:([^\s]+).*?$/)
        {
            my $label = $1;
            my $qid = $2;
            my $did = $lineNum;
            $hsQueryDocLabelScore{$qid}{$did}{"label"} = $label;
            $hsQueryDocLabelScore{$qid}{$did}{"pred"} = $predScore;
            $hsQueryDocLabelScore{$qid}{$did}{"lineNum"} = $lineNum;
        }
        else
        {
            print "Error to parse $fnFeature at line $lineNum:\n$lnFea\n";
            exit -2;
        }
    }
    close(FIN_Feature);
    close(FIN_Pred);
    return %hsQueryDocLabelScore;
}


sub PrecisionAtN
{
    my ($topN, @rates) = @_;
    my @PrecN;
    my $numRelevant = 0;
#   modified by Jun Xu, 2009-4-24.
#   if # retrieved doc <  $topN, the P@N will consider the hole as irrelevant
#    for(my $iPos = 0;  $iPos < $topN && $iPos < $#rates + 1; $iPos ++)
#
    for (my $iPos = 0; $iPos < $topN; $iPos ++)
    {
        my $r;
        if ($iPos < $#rates + 1)
        {
            $r = $rates[$iPos];
        }
        else
        {
            $r = 0;
        }
        $numRelevant ++ if ($hsPrecisionRel{$r} == 1);
        $PrecN[$iPos] = $numRelevant / ($iPos + 1);
    }
    return @PrecN;
}

sub MAP
{
    my @rates = @_;

    my $numRelevant = 0;
    my $avgPrecision = 0.0;
    for(my $iPos = 0; $iPos < $#rates + 1; $iPos ++)
    {
        if ($hsPrecisionRel{$rates[$iPos]} == 1)
        {
            $numRelevant ++;
            $avgPrecision += ($numRelevant / ($iPos + 1));
        }
    }
    return 0.0 if ($numRelevant == 0);
    #return sprintf("%.4f", $avgPrecision / $numRelevant);
    return $avgPrecision / $numRelevant;
}

sub DCG
{
    my ($topN, @rates) = @_;
    my @dcg;
    
    $dcg[0] = $hsNdcgRelScore{$rates[0]};
#   Modified by Jun Xu, 2009-4-24
#   if # retrieved doc <  $topN, the NDCG@N will consider the hole as irrelevant
   # for(my $iPos = 1; $iPos < $topN && $iPos < $#rates + 1; $iPos ++)
#
    for(my $iPos = 1; $iPos < $topN; $iPos ++)
    {
        my $r;
        if ($iPos < $#rates + 1)
        {
            $r = $rates[$iPos];
        }
        else
        {
            $r = 0;
        }
        # line 309 - 316 is wrong, for instance, $iPos = 1 ~~ ndcg@2
        # $iPos is i-1 in the NDCG formula thus should divide log($iPos + 2.0)
        # instead of log($iPos + 1.0)
        # if ($iPos < 2)
        # {
        #     $dcg[$iPos] = $dcg[$iPos - 1] + $hsNdcgRelScore{$r};
        # }
        # else
        # {
        #     $dcg[$iPos] = $dcg[$iPos - 1] + ($hsNdcgRelScore{$r} * log(2.0) / log($iPos + 1.0));
        # }
        $dcg[$iPos] = $dcg[$iPos - 1] + ($hsNdcgRelScore{$r} * log(2.0) / log($iPos + 2.0));
    }
    return @dcg;
}
sub NDCG
{
    my ($topN, @rates) = @_;
    my @ndcg;
    my @dcg = DCG($topN, @rates);
    my @stRates = sort {$hsNdcgRelScore{$b} <=> $hsNdcgRelScore{$a}} @rates;
    my @bestDcg = DCG($topN, @stRates);
    
    # for(my $iPos =0; $iPos < $topN && $iPos < $#rates + 1; $iPos ++)# modified
    for(my $iPos =0; $iPos < $topN; $iPos ++)# modified
    {
        $ndcg[$iPos] = 0;
        $ndcg[$iPos] = $dcg[$iPos] / $bestDcg[$iPos] if ($bestDcg[$iPos] != 0);
    }
    return @ndcg;
}

