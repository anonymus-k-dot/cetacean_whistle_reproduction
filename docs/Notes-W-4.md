# Notes-W-4 — Week 4 Project Logbook: Training Experiments & Evaluation

Markdown version of `C:\Users\welcome\Desktop\Notes-W-4.pdf`. Part A is a word-for-word copy of the page. Part B describes the file and its layout. Part C lists every field that needs filling in. Part D lists what the checklist asks for, plus the places where the template disagrees with itself.

> **State of the document:** this is an **empty template**. Every field is blank, all five checklist boxes are unchecked (`[ ]`), and Section IV contains only headings.

---

## Part A — Full transcription (verbatim)

Wording, punctuation, capitalisation, symbols (`→`, `|`, `________`) and bold/italic styling all match the PDF. The PDF itself uses one font size for everything (see Part B), so the Markdown heading levels here are only for readability.

### **Week 4 Project Logbook: Training Experiments & Evaluation**

#### **I. Hyperparameter Tuning & Dataset Experiments**

*(To be completed during the initial tuning phase)*

- **Hyperparameter Grid / Random Search Matrix:**
  - *Learning Rates tested:*
  - *Batch Sizes tested:*
  - *Best Configuration Identified:*
- **Dataset Ratio & Augmentation Hypotheses:**
  - *50% vs 100% Data Impact:*
  - *Augmentation ON vs OFF Impact:*
- **Ablation Study Breakdown (Removing 1 key component at a time):**
  - *Component 1 removed → Metric change:*
  - *Component 2 removed → Metric change:*

#### **II. Test Results Collection & Deep Analysis**

*(To be completed after running the best checkpoint on the held-out test set)*

- **Evaluation Metrics Computed (vs. Paper's Baseline Table):**
  - *Metric 1 (e.g., SNR, F1):* Your Result \_\_\_\_\_\_\_\_ | Paper Result \_\_\_\_\_\_\_\_
  - *Metric 2 (e.g., PESQ, WER):* Your Result \_\_\_\_\_\_\_\_ | Paper Result \_\_\_\_\_\_\_\_
- **Performance Gap Hypothesis (If your results differ from the paper, why?):**
- **Systematic Failure Case Analysis (Group Review):**
  - *Pattern found in failed test samples:*
  - *Are there data quality issues?*

#### **III. Practical Implementation & Verification Checklist**

(Hands-on execution log for Week 4 based on your workflow plan)

- [ ] **Hyperparameter Tuning:** Conducted grid/random search varying learning rate, batch size, and loss weights; logged every run to identify the best configuration.
- [ ] **Dataset & Augmentation Experiments:** Trained models using 50% vs 100% data ratios and tested with augmentation ON vs OFF to measure model data-hunger.
- [ ] **Best Config Training & Ablation:** Re-trained the model with the top hyperparameter setup for full epochs (50-100); ran 3 ablation variants and built a results table matching the paper's ablation section.
- [ ] **Test Results Collection:** Ran the top checkpoint on the held-out test set; computed all relevant metrics and prepared a results summary CSV.
- [ ] **Model Internal Visualizations:** Generated metric bar charts, confusion matrices, or before/after spectrograms; visualized internals like feature maps, attention maps, or latent space t-SNEs for 5 easy and 5 hard samples.

#### **IV. Implementation Notes & Code Snippets**

- **Experiment / Evaluation Insights:**
- **Core Evaluation Script Snippet:**

*(End of page. There is no content below this point.)*

---

## Part B — The file and its layout

### B.1 File properties

| Property | Value |
|---|---|
| File name | `Notes-W-4.pdf` |
| Location | `C:\Users\welcome\Desktop\` |
| File size | 781,335 bytes (~763 KB) |
| PDF version | 1.7 |
| Page count | **1** |
| Page size | 612 × 792 pt = **8.5 × 11 in (US Letter)**, portrait |
| Embedded title | `Microsoft Word - Running notes template` |
| Author | Balarama Murthy P V N |
| Producer | Microsoft: Print To PDF |
| Created / modified | 2026-10-28, 09:44:34 IST (UTC+05:30); both timestamps are the same |
| Subject / keywords | (empty) |
| Encryption | None |
| Images / drawn graphics | None. The page is text only, with no tables, lines, boxes or pictures. |
| Fonts | 7 embedded TrueType CID subsets (`CIDFont+F1` … `F7`, Identity-H encoding). The glyphs look like Calibri, Word's default font, in regular, **bold**, *italic* and symbol variants. |
| Font sizes | ~11.3 pt for all body text and headings. ~9.3 pt is used only for the bullet glyphs. |

**Origin:** the file was written in Microsoft Word from a document called "Running notes template". It was then printed to PDF with Windows' built-in "Microsoft Print to PDF" printer. "W-4" in the file name means Week 4.

**Copy-paste quirk:** the PDF stores the letter pairs *ti*, *ft*, *tt* and *fi* as single ligature glyphs. If you copy text out of the PDF, these come out as odd characters, for example `EvaluaƟon` (Evaluation), `aŌer` (after), `PaƩern` (Pattern) and `ConﬁguraƟon` (Configuration). The page displays correctly. Part A uses the normal spelling.

### B.2 Layout

- **Text area:** text starts about 1.3 in (93.6 pt) from the left edge and ends about 520 pt across. It runs from 67.6 pt down to 618.5 pt, so roughly the **bottom 2.4 in of the page is empty**. That space sits under Section IV, which suggests it is meant for handwritten or typed notes and code.
- **Same size for every heading:** the title and all four section headings are **bold** at body-text size. Nothing is made larger to mark a heading.
- **Section numbers** are Roman numerals: I, II, III, IV.
- **Bullet levels:**
  - Level 1 uses a solid round bullet (`•`). Every level-1 lead-in is **bold** and ends with a colon.
  - Level 2 uses Word's hollow `o` bullet. Every level-2 label is *italic*. Most end with a colon, but "Are there data quality issues?" ends with a question mark.
- **Section instructions** are written in parentheses on the line under the heading:
  - In Sections I and II they are *italic*.
  - In Section III the instruction is **not** italic, unlike the other two.
  - Section IV has no instruction line.
- **Blanks to fill in:** only the two metric lines in Section II have printed blanks. Each blank is 8 underscores (`________`), and each line has two: *Your Result* and *Paper Result*, separated by a vertical bar `|`. Every other field just ends with a colon and leaves the rest of the line empty.
- **Checkboxes:** Section III uses the literal text `[ ]` (an unchecked box) at the start of each bullet, followed by a **bold** label with a colon and then a regular-weight description. The descriptions are written in the past tense ("Conducted…", "Trained…", "Re-trained…", "Ran…", "Generated…"). Each one is a statement you confirm by ticking the box.
- **Arrows:** both ablation lines use a right arrow (`→`, U+2192) between "Component N removed" and "Metric change".

---

## Part C — Every field to fill in

All 19 items below are empty in the source. That is 14 blank fields in Sections I, II and IV, plus the 5 checkboxes in Section III.

| # | Section | Field (exact label) | What goes in it | When |
|---|---|---|---|---|
| 1 | I | Learning Rates tested: | The list of learning-rate values tried | During the initial tuning phase |
| 2 | I | Batch Sizes tested: | The list of batch sizes tried | During the initial tuning phase |
| 3 | I | Best Configuration Identified: | The winning hyperparameter set | During the initial tuning phase |
| 4 | I | 50% vs 100% Data Impact: | Change in the metric when training on half the data vs all of it | During the initial tuning phase |
| 5 | I | Augmentation ON vs OFF Impact: | Change in the metric with augmentation on vs off | During the initial tuning phase |
| 6 | I | Component 1 removed → Metric change: | Ablation 1: which component was removed, and the change in the metric | During the initial tuning phase |
| 7 | I | Component 2 removed → Metric change: | Ablation 2: which component was removed, and the change in the metric | During the initial tuning phase |
| 8 | II | Metric 1 (e.g., SNR, F1): Your Result ________ \| Paper Result ________ | Name of metric 1, your value, and the paper's value | After testing the best checkpoint on the held-out set |
| 9 | II | Metric 2 (e.g., PESQ, WER): Your Result ________ \| Paper Result ________ | Name of metric 2, your value, and the paper's value | After testing the best checkpoint on the held-out set |
| 10 | II | Performance Gap Hypothesis (If your results differ from the paper, why?): | Why your numbers differ from the paper's (free text) | After testing |
| 11 | II | Pattern found in failed test samples: | What the failed samples have in common | After testing (group review) |
| 12 | II | Are there data quality issues? | Yes or no, with evidence | After testing (group review) |
| 13 | III | [ ] Hyperparameter Tuning | Tick when done | Week 4 execution |
| 14 | III | [ ] Dataset & Augmentation Experiments | Tick when done | Week 4 execution |
| 15 | III | [ ] Best Config Training & Ablation | Tick when done | Week 4 execution |
| 16 | III | [ ] Test Results Collection | Tick when done | Week 4 execution |
| 17 | III | [ ] Model Internal Visualizations | Tick when done | Week 4 execution |
| 18 | IV | Experiment / Evaluation Insights: | Free-text notes | Any time |
| 19 | IV | Core Evaluation Script Snippet: | A code excerpt | Any time |

---

## Part D — What each checklist item requires

### D.1 Concrete requirements in the Section III checklist

1. **Hyperparameter Tuning**
   - Search method: a **grid search or a random search**.
   - Hyperparameters to vary (three of them): **learning rate**, **batch size**, **loss weights**.
   - Log **every** run, so the best configuration can be identified.
2. **Dataset & Augmentation Experiments**
   - Training-data amount: **50% vs 100%** of the data.
   - Augmentation: **ON vs OFF**.
   - Stated purpose: to measure how *data-hungry* the model is.
3. **Best Config Training & Ablation**
   - Retrain with the **top hyperparameter setup**.
   - Train for **full epochs, 50–100**.
   - Run **3 ablation variants**.
   - Build a **results table that matches the paper's ablation section**.
4. **Test Results Collection**
   - Use the **top checkpoint**.
   - Run it on the **held-out test set**.
   - Compute **all relevant metrics**.
   - Deliverable: a **results summary CSV**.
5. **Model Internal Visualizations** (the "or" options are alternatives)
   - Output plots: **metric bar charts**, **confusion matrices**, or **before/after spectrograms**.
   - Model internals: **feature maps**, **attention maps**, or **latent-space t-SNE plots**.
   - Samples: **5 easy + 5 hard = 10**.

### D.2 Deliverables implied by the template

- A log of every tuning run
- The best configuration
- A data-ratio comparison (50% vs 100%)
- An augmentation comparison (ON vs OFF)
- 3 ablation runs and an ablation table in the paper's format
- Test-set metrics for the best checkpoint
- A results summary CSV
- A side-by-side comparison with the paper (your result vs the paper's)
- A written explanation of the performance gap
- A failure-case analysis, reviewed as a group
- Output plots and model-internals plots for 10 samples
- Written insights
- An excerpt from the evaluation script

### D.3 Places where the template disagrees with itself

These are differences between the fields in Sections I–II and the tasks in Section III. They are useful to know when filling the template in.

| Topic | Section I/II field | Section III task | Gap |
|---|---|---|---|
| Hyperparameters searched | Fields only for **learning rates** and **batch sizes** | Also varies **loss weights** | There is no field for the loss weights that were tested |
| Number of ablations | **2** slots (Component 1, Component 2) | **3** ablation variants | One ablation result has nowhere to go in Section I |
| Number of metrics | **2** metric lines | "computed **all relevant** metrics" | Any extra metrics have no line of their own |
| Instruction styling | Section I and II instructions are italic | Section III instruction is upright | Cosmetic only |

### D.4 The example metrics are not for this project

The examples given (**SNR**, **F1**, **PESQ**, **WER**) come from audio enhancement, speech and classification work. PESQ is a speech-quality score and WER is a speech-recognition error rate. Together with "before/after spectrograms", "attention maps" and "the paper's ablation section", this shows the logbook is a **generic template shared across several paper-reproduction projects**. The metrics you enter should be the ones the reproduced paper actually reports.
