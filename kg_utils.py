import pandas as pd
import re
import spacy
import neuralcoref
from tqdm import tqdm

nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)


def extract_triplets(text, title, global_ents_list, verbose=False):
    """Method to extract Subject-Relation-Object triplets for KG construction
    
    Parameters
    ----------
    text : str
        Raw text from Wikipedia article/document
    title : str
        Title of document, used as a default/fallback subject
    global_ents_list : list
        List of domain-specific entities which Spacy tools do not 
        recognize as Named entities or Nouns chunks
    verbose : bool, optional
        Flag for displaying progress bar and verbose output
        
    Returns
    -------
    sro_triplets_df : pd.DataFrame
        Pandas dataframe with S-R-O triplets extracted from document
    """
    # print("\nRaw text:\n", text)
    text = re.sub(r'\n+', '. ', text)
    # print("\nRemove new-line:\n", text)
    text = re.sub(r'\[\d+\]', ' ', text)
    # print("\nRemove reference numbers:\n", text)
    text = re.sub(r'\([^()]*\)', ' ', text)
    # print("\nRemove parenthesis:\n", text)
    text = re.sub(r'(?<=[.,])(?=[^\s0-9])', r' ', text)
    # print("\nFix formatting:\n", text)
    
    # Resolve coreferences with Spacy
    text = nlp(text)
    text = nlp(text._.coref_resolved)
    # print("\nResolve coreferences:\n", text)
        
    # Track (Subject, Relation, Object) triplets
    sro_triplets = []

    # Temp variables to track previous sentence subject and object
    prev_subj = nlp("")[:]
    prev_obj = nlp("")[:]
    default_subj = nlp(title)[:]  # Set default subject: title of document

    sentences = [sent.string.strip() for sent in text.sents]
    for sent in tqdm(sentences):
        prev_obj_end = 0  # Temp pointer to previous object end

        sent = nlp(sent)  # Pass through Spacy pipeline

        # Retokenize to combine Named Entities into single tokens
        ents = list(sent.ents)
        spans = spacy.util.filter_spans(ents)
        with sent.retokenize() as retokenizer:
            [retokenizer.merge(span) for span in spans]

        # Re-build Named Entities by categories
        ents = list(sent.ents)
        main_ents = []  # Named Entities recognised by Spacy (Main)
        addn_ents = []  # Additional named entities (Date/Time/etc.)
        for ent in ents:
            if ent.label_ in ("DATE", "TIME", "MONEY", "QUANTITY"):
                addn_ents.append(ent)
            elif ent.label_ in ("CARDINAL", "ORDINAL", "PERCENT"):
                # Ignore cardinal/ordinal numbers and percentages
                continue
            elif ent.label_ in ("PERSON", "NORP", "FAC", "ORG", 
                                "GPE", "LOC", "PRODUCT", "EVENT", 
                                "WORK_OF_ART", "LAW", "LANGUAGE"):
                main_ents.append(ent)
        # Identidy Domain-specific/global named entities
        global_ents = []
        for tok in sent:
            if tok.text.lower() in global_ents_list:
                global_ents.append(sent[tok.i:tok.i+1])

        # Identify noun chunks besides Named Entities 
        noun_chunks = list(sent.noun_chunks)

        # Identify verbs for forming relations
        verbs = [tok for tok in sent if tok.pos_ == "VERB"]
        
        if verbose:
            print("\n----------\n")
            print("\nSentence:\n", sent)
            print("\nNamed Entities:\n", main_ents)
            print("\nDomain-specific Entities:\n", global_ents)
            print("\nAdditional Entities:\n", addn_ents)
            print("\nNoun spans:\n", noun_chunks)
            print("\nVerbs:\n", verbs)

        for verb in verbs:
            
            # Identify Subject
            subj = None
            # Find leftmost Main Ent to verb
            for ent in main_ents:
                if ent.end > verb.i:
                    break
                elif ent.end > prev_obj_end:
                    subj = ent
                    rel_start = subj.end
            if subj is None:
                # Find leftmost Global Ent to verb
                for ent in global_ents:
                    if ent.end > verb.i:
                        break
                    elif ent.end > prev_obj_end:
                        subj = ent
                        rel_start = subj.end
            if subj is None:
                # Find leftmost noun chunk to verb
                for noun_chunk in noun_chunks:
                    if noun_chunk.end > verb.i:
                        break
                    elif noun_chunk.end > prev_obj_end:
                        subj = noun_chunk
                        rel_start = subj.end
            if subj is None:
                # Find leftmost Additional Ent to verb
                for ent in addn_ents:
                    if ent.end > verb.i:
                        break
                    elif ent.end > prev_obj_end:
                        subj = ent
                        rel_start = subj.end
            if subj is None:
                # If no subject found, assign default subject
                subj = default_subj
                rel_start = verb.i

            ##########

            # Identify Object
            obj = None
            # Find rightmost Main Ent to verb
            for ent in main_ents[::-1]:
                if ent.end <= verb.i:
                    break
                else:
                    obj = ent
                    rel_end = obj.start
            if obj is None:
                # Find rightmost Global Ent to verb
                for ent in global_ents[::-1]:
                    if ent.end <= verb.i:
                        break
                    elif ent.text.lower() != verb.text.lower():  
                        # Additional check for global entity not being verb itself!
                        obj = ent
                        rel_end = obj.start
            if obj is None:
                # Find rightmost noun chunk to verb
                for noun_chunk in noun_chunks[::-1]:
                    if noun_chunk.end <= verb.i:
                        break
                    else:
                        obj = noun_chunk
                        rel_end = obj.start
            if obj is None:
                # Find rightmost Additional Ent to verb
                for ent in addn_ents[::-1]:
                    if ent.end <= verb.i:
                        break
                    else:
                        obj = ent
                        rel_end = obj.start
            if obj is None:
                # If no object found, assign previous subject
                obj = prev_obj
                rel_end = verb.i + 1

            ##########

            # Identify and lemmatized relationship spans around verb token
            triplet = (
                # Subject
                " ".join(tok.text.lower() for tok in subj if  
                         (tok.is_stop == False and tok.is_punct == False)).strip(), 
                # Relationship
                " ".join(tok.lemma_.lower() for tok in sent[rel_start:rel_end] if 
                         (tok == verb or (tok.is_stop == False and tok.is_punct == False))).strip(),
                # Object 
                " ".join(tok.text.lower() for tok in obj if 
                         (tok.is_stop == False and tok.is_punct == False)).strip(), 
            )
            
            # Append valid SRO triplets to list
            if triplet[0] != "" and triplet[1] != "" and triplet[2] != "" and triplet[0] != triplet[2]:
                # Check for duplicate triplets within same sentence 
                if subj == prev_subj and obj == prev_obj:
                    prev_triplet = sro_triplets.pop()
                    # Define relation as the longest relation span among duplicates
                    if len(prev_triplet[1]) > len(triplet[1]):
                        triplet = prev_triplet
                
                sro_triplets.append(triplet)
                if verbose:
                    print("\nS-R-O:\n", subj, "-", relation, "-", obj)

            # Update previous subject and object variables
            prev_subj = subj
            prev_obj = obj
            prev_obj_end = obj.end
    
    # Convert to df
    sro_triplets_df = pd.DataFrame(sro_triplets, columns=['subject', 'relation', 'object'])
    return sro_triplets_df


def merge_duplicate_subjs(triplets, title=None):
    """Helper function to merge duplicate subjects
    
    Duplicate subjects can be extensions/additional words joined to typical subjects,
    e.g. 'bayer ag', 'bayer healthcare', 'bayer pharmaceuticals' --> 'bayer'
    Note that when merging an extended subject (e.g. 'bayer healthcare') 
    back to a subject ('bayer'), we append the extension ('healthcare') 
    to the relation for the triplet and then replace the extended subject with the subject.
    
    Parameters
    ----------
    triplets : pd.DataFrame
        S-R-O triplets dataframe
    
    Returns
    -------
    triplets : pd.DataFrame
        Updated dataframe with merged duplicate subjects
    """
    subjects = sorted(list(triplets.subject.unique()))
    prev_subj = subjects[0]
    for subj in subjects[1:]:
        # TODO Use string edit distance between prev_subj and subj
        if prev_subj in subj:
            # Detect extension in subj compared to prev_subj and append it to relations of rows with subj
            triplets.loc[triplets.subject==subj, 'relation'] = subj.replace(prev_subj, '').strip() + ' ' + triplets[triplets.subject==subj].relation
            # Update subject from subj to prev_subj
            triplets.loc[triplets.subject==subj, 'subject'] = prev_subj
            
        else:
            # Update prev_subj
            prev_subj = subj
    
    return triplets


def prune_infreq_subjects(triplets, threshold=2):
    """Helper function to prune triplets with infrequent subject
    
    Parameters
    ----------
    triplets : pd.DataFrame
        S-R-O triplets dataframe
    threshold : int
        Frequency threshold for pruning
    
    Returns
    -------
    triplets : pd.DataFrame
        Updated dataframw with pruned rows
    """
    # Count unique subjects
    subj_counts = triplets.subject.value_counts()
    # TODO: add more/smarter heuristics for pruning?
    # Drop subjects with counts below threshold
    triplets['subj_count'] = list(subj_counts[triplets.subject])
    triplets.drop(triplets[triplets['subj_count'] < threshold].index, inplace=True)
    triplets = triplets.drop('subj_count', 1)
    return triplets


def prune_infreq_objects(triplets, threshold=2):
    """Helper function to prune triplets with infrequent objects
    
    Parameters
    ----------
    triplets : pd.DataFrame
        S-R-O triplets dataframe
    threshold : int
        Frequency threshold for pruning
    
    Returns
    -------
    triplets : pd.DataFrame
        Updated dataframw with pruned rows
    """
    # Count unique objects
    obj_counts = triplets.object.value_counts()
    # TODO: add more/smarter heuristics for pruning?
    # Drop objects with counts below threshold
    triplets['obj_count'] = list(obj_counts[triplets.object])
    triplets.drop(triplets[triplets['obj_count'] < threshold].index, inplace=True)
    triplets = triplets.drop('obj_count', 1)
    return triplets


def prune_self_loops(triplets):
    """Helper function to prune triplets where subject is the same as object
    """
    triplets.drop(triplets[triplets.subject==triplets.object].index, inplace=True)
    return triplets
