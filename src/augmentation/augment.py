from eda import *
import pandas as pd

# arguments to be parsed from command line
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float,
                help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

# the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join

    output = join(dirname(args.input), 'eda_' + basename(args.input))

# number of augmented sentences to generate per original sentence
num_aug = 9  # default
if args.num_aug:
    num_aug = args.num_aug

# how much to replace each word by synonyms
alpha_sr = 0.1  # default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

# how much to insert new words that are synonyms
alpha_ri = 0.1  # default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

# how much to swap words
alpha_rs = 0.1  # default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

# how much to delete words
alpha_rd = 0.1  # default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
    ap.error('At least one alpha should be greater than zero')


# generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    writer = open(output_file, 'w', encoding='utf8')
    writer.write("comment_id,text\n")
    df = pd.read_csv(train_orig)
    print(f"Generating augmented sentences for {len(df.index)} rows")
    num_new = 0
    for index, row in df.iterrows():
        label = row['comment_id']
        sentence = row['text']
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                            num_aug=num_aug)
        for aug_sentence in aug_sentences:
            num_new += 1
            writer.write(f"{label},{aug_sentence}\n")
    writer.close()
    print(f"Generated {num_new} sentences from {len(df.index)}")
    print(f"Augmented sentences with EDA for {train_orig} to {output_file} with num_aug={num_aug}")


# main function
if __name__ == "__main__":
    # generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd,
            num_aug=num_aug)
