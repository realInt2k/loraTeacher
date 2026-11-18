# Subliminal Learning from Weight-Finetuned Teachers

This is the shared repository for our course project. It holds the code we use, the rules we follow when running experiments, every result we have so far, and the outline of the report. If you are new to the project, read this page first and then `docs/results.md`.

## What the project is about

A language model can pass on a preference to another model through data that never mentions it. In the standard setup, a "teacher" model is told in its system prompt that it loves cats and is then asked to continue lists of random numbers. A "student" model, starting from the same base model, is fine-tuned only on those number lists. Afterwards, when the student is asked for its favorite animal, it says "cat" noticeably more often than before, even though the word never appeared in its training data. This is called subliminal learning.

Almost all earlier work gives the teacher its preference through a system prompt. Our project asks what happens when the preference is trained into the teacher's weights instead, using a LoRA adapter, and how the size (rank) of the teacher's adapter and of the student's adapter affects how much of the preference gets through. We also look at whether the teacher's and student's weight changes point in similar directions, as the original theory suggests they should, and at what else changes in the student's answers besides the target animal.

We use Qwen2.5-7B-Instruct for everything in the main paper.

## How the repository is organized

If you use Claude Code, it reads `CLAUDE.md` and knows where everything goes, so you can simply ask it to add your code or results. The `docs` folder is where the project lives in written form. `docs/standards.md` describes how every experiment must be run and saved, so that results from different people can sit in the same figure. `docs/results.md` collects everything we have found so far, including the pilots that did not work. `docs/experiments.md` lists what has been done and what is still left, with an owner for each item. `docs/paper.md` is the skeleton of the report, section by section, with a note on which result each section depends on. `docs/background.md` summarizes the papers we build on.

The `pipeline` folder will hold the reference pipeline, which is Erell's code, since it is the setup that produced our main results. The `runs` folder holds the output of every experiment in a fixed format, and the `analysis` folder holds the scripts that turn those outputs into tables and figures. The `figures` folder contains the plots we have so far. Finally, `pilots` keeps earlier code and data that are useful for reference but are not part of the main results.

## The few rules that matter

Every result in the main paper comes from the reference pipeline with the settings written in `docs/standards.md`. If an experiment changes a setting, that change is the point of the experiment and it is written down. Every run is saved as one folder in `runs` with its configuration, the raw answers of the student, and a metrics file, so that anyone can redo the analysis later without rerunning anything. Results that were produced differently are still welcome, but they go into the pilots section rather than into the main figures.

The report is due on December 4.
