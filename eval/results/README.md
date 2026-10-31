# Eval results

`python -m eval.<script>` writes timestamped JSON files here (raw files are
gitignored). After running everything:

    python -m eval.summarize

prints the Markdown table for the README "Results" section, built from the
newest result file of each kind. Paste it in, along with the eval-set sizes.
