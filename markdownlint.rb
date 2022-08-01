################################################################################
# Modified using
# https://github.com/jumanjihouse/pre-commit-hooks/blob/master/ci/jumanjistyle.rb
################################################################################


# frozen_string_literal: true

################################################################################
# Style file for markdownlint.
#
# https://github.com/markdownlint/markdownlint/blob/master/docs/configuration.md
#
# This file is referenced by the project `.mdlrc`.
################################################################################

#===============================================================================
# Start with all built-in rules.
# https://github.com/markdownlint/markdownlint/blob/master/docs/RULES.md
all

#===============================================================================
# Override default parameters for some built-in rules.
# https://github.com/markdownlint/markdownlint/blob/master/docs/creating_styles.md#parameters

# Allow both fenced and indented code blocks.
# rule 'MD046', style: ['fenced', 'indented']

# Ignore line length in code blocks.
rule 'MD013', :line_length => 1000
# rule 'MD013', code_blocks: false

#===============================================================================
# Exclude the rules I disagree with.

# IMHO it's easier to read lists like:
# * outmost indent
#   - one indent
#   - second indent
# * Another major bullet
exclude_rule 'MD004' # Unordered list style

# I prefer two blank lines before each heading.
exclude_rule 'MD012' # Multiple consecutive blank lines

# This is not useful for some files such as `CHANGELOG.md`
exclude_rule 'MD024' # Multiple headers with the same content

# I find it necessary to use '<br/>' to force line breaks.
exclude_rule 'MD033' # Inline HTML

# If a page is printed, it helps if the URL is viewable.
exclude_rule 'MD034' # Bare URL used

# Some md files have comments or links at the top of the files.
exclude_rule 'MD041' # First line in file should be a top level header

#===============================================================================
# Exclude rules for pragmatic reasons.

# Either disable this one or MD024 - Multiple headers with the same content.
exclude_rule 'MD036' # Emphasis used instead of a header
