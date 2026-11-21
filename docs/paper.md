# Paper skeleton

This is the outline of the report. For each section it says what the section needs to argue and which result it depends on, so we can see at a glance which parts can be written now and which are waiting for experiments. Experiment numbers refer to `docs/experiments.md`. A result that does not support any section here does not go into the report.

## Title and abstract

Working title: *Subliminal Learning from Weight-Finetuned Teachers*. The abstract is written last, once the results are fixed. It should state the question, the setup in one sentence, the two or three findings we are confident about, and one limitation.

## 1. Introduction

The introduction explains subliminal learning in plain terms with the cat-and-numbers example, says why it matters (traits can pass between models through data that looks harmless, which is a concern whenever models are trained on other models' outputs), and points out the gap we address: almost all earlier work gives the teacher its trait through a system prompt, while in practice traits are more often trained into the weights. It ends with our questions and a short list of findings. This section can be drafted now. Owner: Ege.

## 2. Background and related work

This section introduces the original result of Cloud et al. and their theoretical argument, the finding of Nief et al. that the effect depends on LoRA rank in an inverted-U shape and on the system prompt, the steering-vector explanation of Blank et al. together with their single fine-tuned teacher, and the work of Feng, who shows that the rank curve depends on the learning rate. It then briefly covers LoRA itself and the findings of Shuttleworth et al. on how LoRA updates differ from full fine-tuning. It should also mention Schrodi et al., who show that the signal sits in a few positions and in early layers. `docs/background.md` has the material. This section can be written now. Owner: Finn.

## 3. Method

The method section describes the pipeline from start to finish: how prompted and fine-tuned teachers are built, how number data is generated and filtered, how students are trained, how answers are counted, and which controls and seeds we use. All values come from `docs/standards.md`, and the pipeline's settings are already there, so this section can be written now. It also explains the fine-tuned teacher recipe, including the lesson from the pilots that teachers must be trained on plain answers. Owner: Erell and Zung.

## 4. Results

**4.1 Fine-tuned teachers transmit their preference.** LoRA fine-tuned teachers pass their preference to students through number data for all four animals, and for eagle, owl and wolf more strongly than prompted teachers. The neutral teacher control stays at the base rate. This section needs the teacher-strength table (experiment 4), so the comparison with prompted teachers is fair, and the fine-tuned neutral teacher (experiment 5). Owner: Erell.

**4.2 How teacher rank and student rank matter.** This is the core of the paper. It shows transfer as a function of student rank, with one curve per teacher rank and per-seed points (experiments 2 and 8). The main observation so far is that fine-tuned teachers transfer best at low student rank, while prompted teachers often peak at high rank. The learning-rate sweep (experiment 3) tests whether the low-rank end depends on the learning rate, as Feng found for prompted teachers. Owner: Erell and Ege.

**4.3 Is the transfer specific to the target animal?** This section asks whether students pick up the teacher's animal specifically or reshuffle their answers more generally, using the teacher-by-answer matrix (experiment 6). It includes the rise of panda in the cat students. Nief et al. report only the target animal, so this is one of our own contributions. Owner: Zung.

**4.4 Weight alignment between teacher and student.** This section tests the prediction from Cloud et al.'s theory: within each student rank, do students whose weight update aligns more with their teacher's pick up more of the trait, compared with the students of the neutral teacher (experiment 7)? If they do, it offers one explanation for why rank and learning rate change transfer. It should be presented as exploratory. Owner: Ege.

**4.5 Dependence on the system prompt.** A short section, only if experiment 9 is run. Owner: Erell.

## 5. Discussion

The discussion ties the results together: what the comparison between prompted and fine-tuned teachers says about how the trait is carried, what the rank and learning-rate results suggest, and whether specificity and alignment change how rank curves in earlier work should be read. It does not claim to explain why subliminal learning happens; it reports which predictions held. 

## 6. Limitations

We use one model family and a handful of animals, the number of seeds is small, synonyms of animal names are counted only partially, the learning rate is fixed across ranks, and some fine-tuned teacher settings were chosen rather than tuned. The pilots that did not work, on Gemma and with the earlier pipeline, are mentioned here or in the appendix. 
## Appendix

The appendix holds the full answer distributions for every student, the full settings table, the pilot results from `docs/results.md`, and any extra figures that do not fit in the main text.
