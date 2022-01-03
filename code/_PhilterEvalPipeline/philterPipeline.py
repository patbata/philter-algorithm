# Load libaries
import codecs
import itertools
import os
import subprocess
import sys
import time

import more_itertools as mit
import numpy as np
import pandas as pd
import pydata_google_auth as pgauth
import seaborn as sns
import xml.etree.ElementTree as ET

from codecs import open
from google.cloud import storage
from itertools import chain, groupby
from matplotlib import pyplot as plt

# Import rndPy
from rndPy.plotting import set_theme
set_theme(theme="optum")
plt.rcParams['figure.dpi'] = 150

# Set default font sizes for matplotlib.
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

sns.set_style('whitegrid')

class Philter:
    def __init__(self,
                 annotatedNotes,
                 philteredNotes):
        """
        Compares human annotated notes to philter de-identified notes and
        presents some metrics to describe how good Philter de-identifies 
        notes.

        Parameters
        ----------
        annotatedNotes : str
            Path to the directory full of human annotated notes (XML)
        philteredNotes : str
            Path to the directory of de-identified notes by Philter (TXT)
        """

        # Store directory paths
        self.annotatedNotes = annotatedNotes
        self.philteredNotes = philteredNotes
        
        # Initialize self attributes
        self.phi_tags = ['NAME',
                        'PROFESSION',
                        'LOCATION',
                        'AGE',
                        'DATE',
                        'CONTACT',
                        'ID',
                        'OTHERPHI',
                        'NEGATIVES']
        self.did_keys = ['file', 
                         'did_note', 
                         'did_idx']

    # For de-identification
    def ranges(self,
               i):
        """
        From a list of integers, the function gets the start and end 
        of sets of consecutive integers and returs them as tuples in a list
        Example: [1,2,5,6,7] -> [(1,2),(40,43)]

        Parameters
        ----------
        i : list
            A list of integers to get the start and end of consecutive integers

        Yields
        ------
        b[0][1], b[-1][1] : tuple
            A tuple of start and end of consecutive integers
        """
        for a, b in itertools.groupby(enumerate(i), 
                                      lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield b[0][1], b[-1][1]+1
            
    def asterisk_range(self,
                       s):
        """
        Takes in a string and gets the indices/spans of asterisks
        
        Parameters
        ----------
        s : str
            The string to determine the spans of the asterisks
        
        Returns
        -------
        list(ranges(did)) : list
            List of tuples that indicate the spans of the de-identified
            words, indicated by asterisks
        """
        did = [i for i, x in enumerate(s) if x == '*']
        return list(self.ranges(did))
    
    # Extract philtered notes per file as dictionary
    def extract_philter(self,
                        file):
        """
        Extracts the text of the philtered notes and obtains
        corresponding indices and spans of de-identified 
        words into a DataFrame given the filename.

        Parameters
        ----------
        file : str
            The filename of annotated XML file

        Returns
        -------
        df : DataFrame
            A dataframe that contains all words with tags and spans of 
            annotated words, sorted by span indices
        """
        # Read the de-identified text
        fd = open(self.philteredNotes+file, 'r')
        did = fd.read()
        fd.close()

        # Create dictionary of note and span
        did_results = [file[:-4],
                      did,
                      self.asterisk_range(did)]

        # Create results dataframe
        results = dict(zip(self.did_keys, did_results))

        return results

    # Create dataframe of tag and spans of annotated notes
    def extract_annotate(self,
                         file):
        """
        Extracts the corresponding tags and spans of a word in
        an annotated XML file given its filename.

        Parameters
        ----------
            file : str
                The filename of annotated XML file

        Returns
        -------
            df : DataFrame
                A dataframe that contains all words with 
                tags and spans of annotated words, sorted
                by span indices
        """
        # Demo for one note
        tree = ET.parse(self.annotatedNotes+file)
        root = tree.getroot()

        # Initialize dataframe
        df = pd.DataFrame()

        for tag in self.phi_tags:
            # Get the spans and text for a specific tags in the XML 
            items = [[elem.attrib['spans'], elem.attrib['text']] 
                     for elem in root.iter(tag)]
            # Obtain spans and combine into tuple
            spans = list(chain(*[span[0].split(',') for span in items]))
            spans = [(int(item[0]), int(item[1])) for item in 
                     [span.split('~') for span in spans]
                    ]
            # Obtain words
            words = list(chain(*[words[1].split(' ... ') 
                                 for words in items]))
            # Create a temporary DataFrame for certain tag
            temp = pd.DataFrame(
                    {'SPANS': spans,
                     'WORDS': words,
                     'TAG': [tag]*len(spans)
                    })
            # Concatenate temporary dataframe to main
            df = pd.concat([df,temp], axis = 0)
        return df.sort_values('SPANS').reset_index(drop=True)

    # Creates a dataframe of value counts of PHI tags given a condition
    # like TP, FN, FP, TN
    def file_tagcount(self,
                      series,
                      prefix):
        """
        Creates a dictionary for value counts given a series and a prefix.

        Parameters
        ----------
        series : Series
            A series of value counts that'll be
                             converted to a dictionary
        prefix : str
            The prefix before the PHI tag title

        Returns
        -------
        tagcount : dict
            A dictionary of value counts
        """
        # Categories present in series
        tagcount = dict(zip([prefix+ i for i in series.index],
                            series))

        # Categories not present in series
        tagcount_none = [prefix + tag for tag in self.phi_tags 
                         if tag not in series.index]
        tagcount.update(dict(zip(tagcount_none,[0]*len(tagcount_none))))
        return tagcount
    
    # Function for bar counts
    def hcount(self,
               ax,
               dist = None,
               perc = False,
               total = None):
        """
        This function adds count (or percentage) to the tip of the bar 
        in a horizontal bar plot. This should be placed immediately after 
        making the plot. Set the plot to variable ax.

        Parameters
        ----------
        ax : axes
            Indicate which axis the counts should be in a subplot/plot.
        dist : float, optional
            If this is specified, it gives the distance of number to bar.
            When this isn't specified, the default is set to
            7% of original total x-axis.
        perc : bool, optional
            If true, it annotates the percentage instead of counts 
            in plot. You need to determine the total parameter to use this.
        total : int, optional 
            The total used to calculate percentage if perc is specified.
        """
        xlim = ax.get_xlim()
        if dist is None:
            dist = (xlim[1]-xlim[0])*0.07
        for p in ax.patches:
            height = p.get_width()
            if np.isnan(height):
                height = 0
            if perc == True:
                ax.text(height + dist,p.get_y()+p.get_height()/2,
                        '{:.2f}%'.format(100*height/total),
                        ha="center",
                        va="center")
            else:
                ax.text(height + dist,p.get_y()+p.get_height()/2,
                        '{:,.0f}'.format(height),
                        ha="center",
                        va="center")
        ax.set_xlim((xlim[0],xlim[1]*1.12))


    # Get attributes per note
    def note_metrics(self,
                     file,
                     showStrictFN = False):
        """
        Creates a list of metrics comparing Philter
        and human annotations given a filename

        Parameters:
        ----------
        file : str
            The filename of annotated XML file in annotatedPath
        showStrictFN : bool, optional
            If True, this prints the DataFrame containing words
            that were marked as False Negatives due to one of the 
            items in a proper noun not being counted as a TN

        Returns
        ----------
        final_results : dict
            A dictionary of metrics for the specified note
        """
        # Get annotation dataframe
        df = self.extract_annotate(file)

        # Exact spans of philter
        did_idx = self.results.loc[self.results.file == 
                                   file[:-4], 'did_idx'].tolist()[0]
        df_philter = df[df.SPANS.isin(did_idx)]

        # Get missed values due to mismatched length
        for val, i in enumerate(['first', 'last']):
            globals()[i] = df[df.SPANS.apply(
                            lambda x: x[val] in[span[val] for span in did_idx 
                                                if span not in 
                                                df_philter.SPANS.tolist()])]
        # Combine first and last
        missing = (pd.concat([first,
                             last],
                            axis = 0)
                    .drop_duplicates())

        # Concat missing with detected
        df_philter = pd.concat([df_philter,
                               missing],
                              axis = 0)
        df_philter['PHILTER'] = 'YES'

        # Words that were not philtered
        df_unan = df[~df.index.isin(df_philter.index)]
        df_unan.insert(3, 'PHILTER', 'NO')

        # Concat unphiltered and philtered
        df_metrics = (pd.concat([df_philter,
                                df_unan],
                                axis = 0)
                        .sort_index())

        # Be strict with philtered word counts (proper nouns)
        # Temporary dataframe to append changes
        df_combo = df_metrics[(df_metrics.TAG != 'NEGATIVES')]

        # Concat start and end of span
        df_combo = pd.concat([df_combo,
                            pd.DataFrame(df_combo['SPANS'].tolist(),
                                         index=df_combo.index,
                                         columns = ['START', 'END'])],
                           axis = 1)

        # Create criteria columns
        df_combo.loc[:,'DIFF'] = df_combo['START'].shift(-1) - df_combo.END
        df_combo.loc[:,'NEXT'] = df_combo['TAG'].shift(-1)

        # First row of noun combination
        word_ind = df_combo[(df_combo.TAG == df_combo.NEXT) &
                (df_combo.DIFF <= 2)].index

        # Get the next row that meets criteria
        f1 = lambda x: x
        f2 = lambda x: x+1
        word_ind = list(set([f(x) for x in word_ind for f in (f1,f2)]))

        # Groupby based on TAG value in df_combo
        tag_ind = df_combo.loc[word_ind, 'TAG'].tolist()
        tag_ind = [list(g) for k, g in groupby(tag_ind)]

        # Store values that met and didn't meet the strict criteria
        df_tp = pd.DataFrame()
        df_fn = pd.DataFrame()

        # Extract indices using tag_ind
        idx = 0
        for val,i in enumerate(tag_ind):
            # Replace TAG with index
            combo_ind = (word_ind[idx:idx+len(i)])
            # Slide forward
            idx += len(i)
            # Replace based on strict criteria
            if all(df_combo.loc[combo_ind,'PHILTER'] == 'YES'):
                tp_temp = df_combo.loc[combo_ind,df_metrics.columns]
                tp_temp.loc[:,'GROUP'] = val
                df_tp = pd.concat([df_tp,
                                  tp_temp],
                                 axis = 0)
                df_combo.loc[combo_ind, 'PHILTER'] = 'YES'
            else:
                fn_temp = df_combo.loc[combo_ind,df_metrics.columns]
                fn_temp.loc[:,'GROUP'] = val
                df_fn = pd.concat([df_fn,
                                  fn_temp],
                                 axis = 0)
                df_combo.loc[combo_ind, 'PHILTER'] = 'NO'

        # Combine new dataframe to df_metrics
        df_metrics = (pd.concat([df_metrics,
                                 df_combo.loc[:, df_metrics.columns]],
                                 axis = 0)
                        .drop_duplicates(keep = 'last', subset='SPANS')
                     )
        
        # TN, FP, TN, FN
        tp = sum((df_metrics.TAG != 'NEGATIVES') & 
                 (df_metrics.PHILTER == 'YES'))
        fp = sum((df_metrics.TAG == 'NEGATIVES') & 
                 (df_metrics.PHILTER == 'YES'))

        # Count of those un-philtered
        tn = sum((df_metrics.TAG == 'NEGATIVES') & 
                 (df_metrics.PHILTER == 'NO'))
        fn = sum((df_metrics.TAG != 'NEGATIVES') & 
                 (df_metrics.PHILTER == 'NO'))

        # Initialize results
        final_results = {'file':file[:-4],
                        'tp':tp,
                        'fp':fp,
                        'tn': tn,
                        'fn':fn,
                        }

        # TPR values
        tpr = (df_metrics[(df_metrics.TAG != 'NEGATIVES') & 
                         (df_metrics.PHILTER == 'YES')]
                    .TAG
                    .value_counts())
        # FNR values
        fnr = (df_metrics[(df_metrics.TAG != 'NEGATIVES') & 
                     (df_metrics.PHILTER == 'NO')]
                    .TAG
                    .value_counts())
        # Strict False Negatives
        if len(df_fn) == 0:
            tpfn = pd.DataFrame()
        else:
            tpfn = (df_fn.groupby('GROUP')
                           .first()
                           .TAG
                           .value_counts())

        # Get dictionary of values 
        metric_type = [tpr, fnr, tpfn]
        for val, i in enumerate(['tpr', 'fnr', 'tpfn']):
            final_results.update(self.file_tagcount(metric_type[val],i))

        if showStrictFN:
            print(df_fn.to_markdown())
            return final_results
        else:
            return final_results
    
    def obtainMetrics(self):
    # Extract all results from philter
        self.results = pd.DataFrame(map(self.extract_philter, 
                                   os.listdir(self.philteredNotes)
                                        )
                                    )
        # Create metrics dataframe from annotation
        self.dfMetrics = pd.DataFrame(map(self.note_metrics, 
                                      os.listdir(self.annotatedNotes)
                                          )
                                      ).set_index('file')
        # Metric summary
        self.metricSummary = self.dfMetrics.sum(axis = 0)
        self.tp, self.fp, self.tn, self.fn = self.metricSummary[0:4].values
        # TPR, recall, sensitivity
        self.sens = self.tp/(self.tp+self.fn) 
        # TNR, specificity, selectivity
        self.spec = self.tn/(self.fp+self.tn) 
        # Precision, positive predictive value
        self.prec = self.tp/(self.tp+self.fp) 
        # harmonic mean of precision & sensitivity
        self.f1 = (2*self.tp)/((2*self.tp)+self.fp+self.fn) 
        # weighted average of recall and precision; values recall
        self.f2 = (5 * self.prec * self.sens)/((4*self.prec)+self.sens) 
        #balanced accuracy
        self.balacc = (self.sens + self.spec)/2  
    
    # Print outputs and results
    def PrintOutputs(self, figsize=(10,8)):
        """
        Print the final metrics comparing the human and computer 
        annotated notes. 
        
        Parameters
        ----------
        figsize : tuple, optional
            A tuple of two values corresponding to the width
            and height of the plot, respectively.
        """
        # Print Outputs
        print(f'{"Sensitivity/True Positive Rate:":<36} {self.sens:>35,.4f}')
        print(f'{"Specificity/True Negative Rate:":<36} {self.spec:>35,.4f}')
        print(f'{"Precision/Positive Predictive Value:":<35} {self.prec:>35,.4f}')
        print(f'{"F1 Score:":<36} {self.f1:>35,.4f}')
        print(f'{"F2 Score:":<36} {self.f2:>35,.4f}')
        print(f'{"Balanced Accuracy:":<36} {self.balacc:>35,.4f}')

        # Initialize result plots
        fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        # Confusion Matrix
        sns.heatmap(
            pd.DataFrame([[self.tn,self.fp],[self.fn,self.tp]], 
                         *[['negative', 'positive']]*2),
            cmap='YlOrRd', 
            annot=True,
            ax=ax1,
            fmt='d'
        )
        ax1.set(title = 'Confusion Matrix\nPhiltered Words',
               ylabel = 'Actual',
               xlabel = 'Predicted')

        # True positive categories
        tpr = self.metricSummary[(self.metricSummary.index.str.startswith('tpr')) &
                             (self.metricSummary.values > 0)]\
                        .sort_values(ascending=False)
        sns.barplot(y=[title[3:] for title in tpr.index], 
                    x=tpr,
                    ax=ax2)
        ax2.set(title = "Count of PHI Categories Philtered\n(True Positive)",
              ylabel= "PHI Category",
              xlabel = "Counts")
        self.hcount(ax=ax2)

        # False negative categories
        fnr = self.metricSummary[(self.metricSummary.index.str.startswith('fnr')) &
                            (self.metricSummary.values > 0)]\
                    .sort_values(ascending = False)
        sns.barplot(y=[title[3:] for title in fnr.index], 
                    x=fnr,
                    ax=ax3)
        ax3.set(title = "Count of PHI Categories Philtered\n(False Negative)",
              ylabel= "PHI Category",
              xlabel = "Counts")
        self.hcount(ax=ax3)

        # Words that are false negatives due to part of the word being false negative
        tpfn = self.metricSummary[(self.metricSummary.index.str.startswith('tpfn')) &
                            (self.metricSummary.values > 0)]\
                    .sort_values(ascending = False)
        sns.barplot(y=[title[4:] for title in tpfn.index],
                    x=tpfn,
                    ax=ax4)
        ax4.set(title = "Count of PHI Categories with Parts Un-Philtered\n(Strict False Negative)",
              ylabel= "PHI Category",
              xlabel = "Counts")
        self.hcount(ax=ax4)

        plt.tight_layout()      
        
        
        