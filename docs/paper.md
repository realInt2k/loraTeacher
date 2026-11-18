# Paper skeleton

This is the outline of the report. For each section it says what the section needs to argue and which result it depends on, so we can see at a glance which parts can be written now and which are waiting for experiments. Experiment numbers refer to `docs/experiments.md`. A result that does not support any section here does not go into the report.

## Title and abstract

Working title: *Subliminal Learning from Weight-Finetuned Teachers*. The abstract is written last, once the results are fixed. It should state the question, the setup in one sentence, the two or three findings we are confident about, and one limitation.

## 1. Introduction

The introduction explains subliminal learning in plain terms with the cat-and-numbers example, says why it matters (traits can pass between models through data that looks harmless, which is a concern whenever models are trained on other models' outputs), and points out the gap we address: almost all earlier work gives the teacher its trait through a system prompt, while in practice traits are more often trained into the weights. It ends with our questions and a short list of findings. This section can be drafted now. Owner: Ege.

## 2. Background and related work

This section introduces the original result of Cloud et al. and their theoretical argument, the finding of Nief et al. that the effect depends on LoRA rank in an inverted-U shape and on the system prompt, the steering-vector explanation of Blank et al. together with their single fine-tuned teacher, and the work of Feng, who shows that the rank curve depends on the learning rate. It then briefly covers LoRA itself and the findings of Shuttleworth et al. on how LoRA updates differ from full fine-tuning. `docs/background.md` has the material. This section can be written now. Owner: Finn.

## 3. Method

The method section describes the pipeline from start to finish: how prompted and fine-tuned teachers are built, how number data is generated and filtered, how students are trained, how answers are counted, and which controls and seeds we use. All values come from `docs/standards.md`, so this section can be written as soon as Erell's settings are confirmed (experiment 1). It also explains the fine-tuned teacher recipe, including the lesson from the pilots that teachers must be trained on plain answers. Owner: Erell and Zung.

## 4. Results

**4.1 Fine-tuned teachers transmit their preference.** The first result is that LoRA fine-tuned teachers pass their preference to students through number data, for all four animals, and often more strongly than prompted teachers. This section needs the teacher strength table (experiment 4) so that the comparison with prompted teachers is fair, and the control teachers and students (experiment 3) to show that the effect comes from the preference. Owner: Erell.

**4.2 How teacher rank and student rank matter.** This is the core of the paper. It shows transfer as a function of student rank, separately for each teacher rank, with confidence intervals over seeds (experiments 2, 5 and 7). It should say plainly where we see an inverted U and where we do not, and note that a fixed learning rate across student ranks is a known confound. Owner: Erell and Ege.

**4.3 What else changes in the student.** This section looks beyond the target animal at the full answer distribution. It presents the panda shift at low rank and whether it holds across teachers and animals, and how the spread of answers changes with rank (experiment 8). Nief et al. do not report this, so it is one of our own contributions. Owner: Finn.

**4.4 Weight alignment between teacher and student.** This section tests the prediction from Cloud et al.'s theory, comparing teacher-student similarity with the similarity seen for controls and other seeds, and relating it to how much each student picked up the preference (experiment 9). It should be presented as exploratory. Owner: Ege.

**4.5 Inside the teacher.** If experiment 10 gives a clear picture, this section shows which tokens the teacher's middle layers favor while writing numbers, compared with the base model, for prompted and fine-tuned teachers. Owner: Zung.

**4.6 Dependence on the system prompt.** A short section with the results of re-evaluating students under a different system prompt (experiment 6). Owner: Erell or Zung.

## 5. Discussion

The discussion ties the results together: what the comparison between prompted and fine-tuned teachers says about how the trait is carried, what the rank results suggest, and how the panda shift changes the way rank curves in earlier work should be read. 

## 6. Limitations

We use one model family and a handful of animals, the number of seeds is small, synonyms of animal names are counted only partially, the learning rate is fixed across ranks, and some fine-tuned teacher settings were chosen rather than tuned. The pilots that did not work, on Gemma and with the earlier pipeline, are mentioned here or in the appendix. 
## Appendix

The appendix holds the full answer distributions for every student, the full settings table, the pilot results from `docs/results.md`, and any extra figures that do not fit in the main text.
