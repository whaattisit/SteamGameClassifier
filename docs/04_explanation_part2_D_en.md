# Part 2 (section D: Text Mining) in simple words

This file explains `notebooks/04_text_mining.ipynb`, which is Part 2 of the assignment. The assignment says "Select the section that matches the project". Our project type is **Text Mining** (Part 1, section 1.5), so we do **section D**. The section numbers below match the notebook.

**The simple question:** what is the best way to turn review text into numbers, how well do the words alone predict 👍/👎, and **why** does the model make mistakes?

This file contains no numbers on purpose, because the notebook computes them. Cells marked **Findings** print the interpretation as sentences built from the numbers of the current run, so the text always matches the results. Copy those lines into the report and the slides. The last cell (`summary`) prints the final conclusions and the answer to RQ1.

---

## How to run

1. Put `04_text_mining.ipynb` into `notebooks/` next to `01`, `02` and `03`.
2. Notebooks `01` and `03` must have been run first. They create `data/processed/*_features.parquet` and `models/preprocessor_tabular.joblib`.
3. Run *Kernel → Restart & Run All*. No new libraries are needed, and it works with Python 3.9 and scikit-learn 1.5. It takes about 20–35 minutes.
4. Save the notebook **with outputs**, so the tables and charts are visible on GitHub.
5. Read every *Findings* line. If the result differs from what you expected, write what the data shows, not the textbook expectation.

Figures go to `reports/figures/` (D01…D13), tables to `reports/tables/` (D_…), and the final model to `models/D_final_text_model.joblib`.

---

## Common rules (assignment section 6)

Learn these rules. You will definitely be asked about them at the defense.

| Rule | In simple words |
|---|---|
| **train** | the data the model learns from |
| **validation** | used to choose the best settings, the best model and the threshold (the "practice exam") |
| **test** | opened **once**, at the very end (the "real exam") |
| **baseline** | the dumbest possible model; a real model must beat it by a clear margin |
| **hyper-parameter tuning** | try several settings (e.g. C = 0.3, 1, 3, 10) and keep the best one on validation |
| **overfitting check** | compare the score on train and validation (validation curves, learning curve); a big gap means the model memorised train |
| **choosing the final model** | the best on validation; if two are almost equal (< 0.003), take the simpler one |

**Why not accuracy?** About 89% of reviews are 👍. A model that **always** says 👍 gets 89% accuracy and finds zero 👎 reviews. That is why the main metric is **F1-macro**.

| Metric | In simple words |
|---|---|
| **recall (👎)** | what share of all negative reviews the model found |
| **precision (👎)** | of the reviews the model called 👎, what share really are 👎 |
| **F1** | one number combining precision and recall (their harmonic mean) |
| **F1-macro** | the average of F1 for 👍 and for 👎; the rare class counts as much as the common one |
| **F1-macro (tuned)** | F1-macro when every model gets its own best threshold (explained in D.4) |
| **PR-AUC** | model quality across **all** thresholds at once; a random model gets the share of 👎 (~0.11) |
| **ROC-AUC** | the probability that a random 👎 gets a more "negative" score than a random 👍; 0.5 = random |
| **confusion matrix** | a 2×2 table: how many 👎 were found, how many were missed, how many false alarms |

---

## D. Text Mining (notebook 04)

### D.1 Text preprocessing
- **Cleaning** was already done in notebook 01: lower case, no links or tags.
- **Tokenisation.** The default sklearn pattern keeps tokens of 2+ characters, so it drops *i* and *a* and cuts *don't* → *don*. An alternative pattern keeps *don't* and *isn't* whole. Validation decides which is better.
- **Stop words.** The standard English list contains *not, no, never, nothing*. A table shows how much more often these words appear in 👎 reviews, so we also test a **custom list** without negations. Note that sklearn's list has no contractions, so *don't* survives either way.
- **No stemming.** There is plenty of data, bigrams capture phrases, and stemmers make mistakes (Porter turns *universe* and *university* into the same *univers*). Stemming is listed as a future improvement.
- **Zipf's law.** A few words are very frequent and most are rare. `min_df=5` cuts off the tail of typos.

### D.2 Bag-of-Words vs TF-IDF
- **BoW**: one column per word; the value is how many times the word appears (or 0/1). Word order is lost, hence the "bag".
- **TF-IDF**: words that are frequent everywhere (*game*) get a small weight; rare but characteristic words (*refund*) get a large one.
- **Bigrams**: word pairs (*not worth*, *highly recommend*) bring back part of the word order.

We compare 8 variants with **the same** classifier (logistic regression), so only the text representation differs. If two scores are almost equal (difference < 0.002), the variant with the smaller vocabulary wins.

**Robustness check:** every variant is also scored with Naive Bayes. If NB ranks the chosen representation near the top too, the choice does not depend on the classifier. If not, the notebook says so, and it goes into the limitations.

The **Findings** table turns the comparison into sentences such as "adding bigrams: helps (+…)" and "standard stop list: hurts (−…)". Use these, not general expectations.

### D.3 Cosine similarity
**Cosine** measures the angle between the vectors of two reviews: 1 means the same words in the same proportions, 0 means nothing in common. Review length does not matter. Rows are normalised, so cosine is just the dot product, and millions of similarities come from one matrix multiplication.

Four uses:
1. **Search**: a query like "game keeps crashing after the update" returns the most similar reviews. A support team would use this to find all complaints about one problem.
2. **Near-duplicates and label noise**: for each validation review we find the most similar train review. Identical texts are short generic reviews from different players, not copied rows (those were removed in notebook 01). The notebook also counts how often the **same text has opposite labels** (for example, "good game" with 👎). No text-only model can get those right, so they set a "ceiling" on accuracy.
3. **Class centroids**: the "average 👎" and the "average 👍" review. The rule "which centroid is closer" is a separate model (Rocchio).
4. **Game similarity**: a heatmap of which games are described with the same words.

### D.4 Models

| Model | Idea |
|---|---|
| Nearest centroid (cosine) | is the review closer to the "average 👎" or the "average 👍"? |
| Complement Naive Bayes | the probability of every word in every class; a variant designed for imbalanced text |
| Logistic regression | sum of word weights → probability |
| Linear SVM | the widest gap between the classes |

**Class imbalance** is handled in two ways:
1. LR and SVM use `class_weight="balanced"`, so a mistake on 👎 costs ~8 times more. This has the same effect as oversampling, but no data is thrown away or copied. Validation and test stay "as in real life".
2. **Threshold moving.** Naive Bayes and the centroid model have no class weights, so comparing them at the default threshold would be unfair. Each model therefore gets its own best threshold. To keep this honest, the threshold is chosen on one half of validation and scored on the other half, then the halves swap (**cross-fitting**). This gives **F1-macro (tuned)**, which is used to choose the hyper-parameters and the final model.

Then come the comparison table, precision–recall curves, validation curves, an overfitting check (with an automatic warning if the best setting is at the edge of the grid), the final model choice, threshold tuning, a learning curve, the text + tabular check, **test once**, the bootstrap 95% CI and the confusion matrix.

**Final model rule:** the best F1-macro (tuned); if a simpler model is within 0.003, take the simpler one (order: centroid → NB → LR → SVM). The notebook prints a note when this happens.

**Learning curve (D.4.3):** the final model is trained on 10%, 25%, 50% and 100% of the training data. If validation is still rising at 100%, more data would help. A big, stable gap between train and validation means overfitting.

**Text + tabular (D.4.4):** section 1.5 mentions tabular features (playtime, author info, game) as a secondary input. This cell adds them to the text and compares on **validation only**. It shows with a number whether the words alone are enough, without touching the test set.

### D.5 Which words decide
The most "negative" and most "positive" words and bigrams of the final model. A Findings line counts how many of the top negative words contain a negation, which is the signal the standard stop-word list would delete.

### D.6 Error analysis (a required part of the assignment)
Two kinds of errors:
- **missed negative**: the author gave 👎, the model said 👍;
- **false alarm**: the author gave 👍, the model said 👎.

What the notebook does:
1. Shows the **10 most confident errors** of each kind with the original text (D.6.1).
2. Computes the error rate by **review length** and by **game** (D.6.2).
3. Computes **patterns** with a "risk ratio", which says how many times more often an error happens when the pattern is present (D.6.3):
   - mixed opinion (*but, however*);
   - sarcasm and jokes (*10/10, lol, /s*);
   - negations;
   - very short reviews;
   - no strongly "coloured" word at all.
4. Prints **real test-set examples for each pattern** and a list of **label-noise candidates** (the errors the model is most sure about) (D.6.4). Use these real examples in the presentation and do not invent your own.
5. **Breaks down individual errors word by word**: which words pulled towards 👎 and which towards 👍 (D.6.5).
6. Shows **successful predictions** (D.6.6), as section 8 asks: the most confident correct 👎 and 👍 reviews, a word-by-word explanation, and a table comparing correct and wrong reviews (length, polar words, "but", negations).

Typical reasons for errors. These are hypotheses that the notebook tests on our data:

| Reason | What it looks like |
|---|---|
| sarcasm | positive words, negative meaning |
| "good, **but**…" | praise and criticism in one review |
| complaints with 👍 | lists problems, still recommends |
| too short | one or two words, no opinion |
| label noise | the text contradicts the thumb (mis-click, edited review) |

**Key point for the defense:** Bag-of-Words adds up word weights and does not understand context. That is why sarcasm and "…, but …" cause errors (check the risk ratios). A model that reads word order (BERT) could fix this, and that is a future improvement. Label noise cannot be fixed by any text model.

### D.7 Conclusions
The `summary` cell prints the results and the **answer to RQ1**. The answer is **not hard-coded**. It is "YES" only if the 95% bootstrap interval of the gain over the baseline is entirely above zero for both F1-macro and PR-AUC. This is a **paired** bootstrap: the model and the baseline are scored on the same resampled reviews. That is what "significantly better" in RQ1 means.

Below the summary, the notebook lists the limitations and 5 future improvements: ready-made text for section 8 of the report, which needs at least 3.

---

## Mini-glossary for Part 2

| Term | In simple words |
|---|---|
| Baseline | the simplest model, the one you must beat |
| Hyper-parameter | a model setting that we choose (C, alpha), not something the model learns |
| Regularisation (C) | a penalty for weights that are too large; small C = stronger penalty = simpler model |
| Validation curve | a chart of score vs hyper-parameter, for train and validation |
| Learning curve | a chart of score vs amount of training data; shows overfitting and whether more data helps |
| Threshold moving | moving the 👍/👎 border from 0.5 to the best value |
| Cross-fitting | choose something on one half of the data, score it on the other half, swap; gives an honest number |
| Bootstrap CI | the spread of a metric when the test set is resampled with replacement many times |
| Paired bootstrap | model and baseline are scored on the same resampled data, so their difference is fair |
| Centroid | the "average" review of a group (e.g. the average 👎 review) |
| Cosine similarity | how similar two texts are, from 0 to 1, independent of length |
| Label noise | reviews whose label does not match their text (joke, mis-click) |
| Risk ratio | how many times more often an error happens when a pattern is present |

---

## Questions you may be asked at the defense

**Why is F1-macro the main metric and not accuracy?**
The "always 👍" model has 89% accuracy and is useless. F1-macro gives the rare 👎 class the same weight as the common one.

**How did you deal with the class imbalance?**
Class weights for LR and SVM: the same effect as resampling, without losing or copying data. Every model also gets its own tuned threshold. Validation and test were not balanced, because they must look like real life.

**Why did you compare models with a tuned threshold?**
Naive Bayes and the centroid model have no class weights, so at the default threshold they look worse than they really are. The final model is used with a tuned threshold anyway, so we compare all models the way they will actually be used. The threshold is chosen on one half of validation and scored on the other, so the number is not optimistic.

**How did you check that the model is not overfitted?**
We compared train and validation F1 for every model, built validation curves and a learning curve, and checked that the best hyper-parameter is not at the edge of the grid. The test set was used once; compare the validation and test numbers printed in D.4.5.

**Why did you choose a slightly worse model?** *(If the notebook prints the "Note".)*
We declared the rule before seeing the results: differences under 0.003 F1 are smaller than the noise between two random validation sets, and a simpler model is faster, easier to explain and less likely to overfit.

**Why Text Mining? Why does the final model use only text?**
The main input is the review text; this was justified in Part 1 (1.5). D.4.4 (and notebook 03) measure what the tabular features add on top of the text on validation. Quote that number.

**Why is the final model linear?**
*(If that is how it turns out.)* On text, linear models are usually the best: text is many weak word signals, and a linear model adds them up. They train in seconds and are explainable (you can see the weight of every word).

**Why does the model get some reviews wrong?**
Bag-of-Words does not understand context: sarcasm, "good, but…", complaints with 👍. Some errors are label noise: the same text appears with both 👍 and 👎. This is shown in D.3 and D.6. Quote the risk ratios and the label-noise share.

**Is RQ1 answered "significantly"?**
Yes, if the paired bootstrap 95% interval of the gain over the baseline does not include zero. The notebook checks this automatically.

**Where did you use cosine similarity?**
Searching for similar reviews, checking near-duplicates and label noise between train and validation, the nearest-centroid classifier, and game similarity (D.3).

**Which three improvements do you suggest?**
1) A transformer (DistilBERT) to understand context and sarcasm; 2) more data (we use a 2% sample; the learning curve shows whether it helps) and other languages; 3) character n-grams for typos and slang. Also: label-noise handling and testing stemming.
