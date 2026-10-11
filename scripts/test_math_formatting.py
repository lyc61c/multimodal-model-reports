"""Regression contracts for GitHub math formatting (standard library only).

Run with ``python -m unittest discover -s scripts -p test_math_formatting.py -v``.
Fixtures retain the TeX that GitHub's Markdown stage previously changed. These
tests check source preservation and the observed HTML-escaping boundary; the
separate MathJax verifier is responsible for actually compiling TeX.
"""
from __future__ import annotations

import html
import re
import unittest
from html.parser import HTMLParser

from format_articles import math_expressions, normalize_math_details


class _MathText(HTMLParser):
    """Read one HTML serialization layer, as DOM textContent does."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def observed_github_inline_text(tex):
    """Model the two escapes observed in published protected-inline math.

    The published HTML contained ``y_{&amp;lt;t}``, so the browser's single
    HTML parse produced ``y_{&lt;t}``, rather than the original ``y_{<t}``.
    Do not repair this with repeated unescape: avoid angle brackets in the
    protected-inline TeX in the first place.
    """
    encoded = html.escape(html.escape(tex, quote=False), quote=False)
    parser = _MathText()
    parser.feed("<math-renderer>" + encoded + "</math-renderer>")
    return "".join(parser.parts)


class DisplayMathRegressionTests(unittest.TestCase):
    def assert_fenced(self, tex, source=None):
        source = "$$\n" + tex + "\n$$\n" if source is None else source
        normalized, _ = normalize_math_details(source)
        self.assertNotIn("$$", normalized)
        self.assertRegex(normalized, r"(?m)^```math\s*$")
        self.assertEqual(
            list(math_expressions(normalized)),
            [{"tex": tex.strip(), "display": True, "delimiter": "math_fence"}],
        )
        self.assertEqual(normalize_math_details(normalized)[0], normalized)
        return normalized

    def test_sets_and_negative_thin_space_are_literal(self):
        # SliME set braces and Qwen2.5-VL's \! were consumed as Markdown escapes.
        cases = [
            r"k=\min\left\{r:\sum_{i=1}^{r}p_i>\gamma\right\}.",
            r"\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.",
            r"\{x_i:x_i<0\}\subseteq\mathcal X,\qquad x\ne y.",
        ]
        for tex in cases:
            with self.subTest(tex=tex):
                self.assert_fenced(tex)

    def test_palm_e_cases_preserve_row_break_and_optional_spacing(self):
        # papers/palm-e/arxiv-2303.03378/README.md, section 4.
        tex = (
            r"x_i=\begin{cases}\gamma(w_i),&\text{文字位置},\\[2pt]"
            r"\phi_j(O_j)_i,&\text{观测位置}.\end{cases}"
        )
        result = self.assert_fenced(tex, "$$" + tex + "$$")
        self.assertIn(r"\\[2pt]", result)

    def test_qwen_rotation_matrix_preserves_two_backslashes(self):
        # papers/qwen/qwen2.5-vl/README.md, section 4.2.
        tex = (
            "R_\\omega(p)=\n\\begin{bmatrix}\n"
            r"\cos(p\omega)&-\sin(p\omega)\\" + "\n"
            r"\sin(p\omega)&\cos(p\omega)" + "\n\\end{bmatrix}."
        )
        self.assert_fenced(tex)

    def test_stars_and_underscores_do_not_become_emphasis(self):
        for tex in (
            r"a_*+b_*=c_*,\qquad x_{a_b}^*=y^*.",
            r"\theta^*=\arg\min_\theta\mathcal L(\theta),\qquad h_i*h_j.",
            r"x_{\mathrm{token}}=a_1+b_2,\qquad a\_b.",
        ):
            with self.subTest(tex=tex):
                self.assert_fenced(tex)

    def test_leo_multiline_expression_is_not_lost(self):
        # The real omission split before a denominator beginning with '{'.
        # Also preserve the loss whose opening $$ shares a line with TeX.
        # papers/leo/arxiv-2311.12871/README.md, section 4.
        cases = [
            (
                r"\omega_{ij}=\frac{\sigma(\omega^s_{ij})\exp(\omega^o_{ij})}"
                + "\n" + r"{\sum_l\sigma(\omega^s_{il})\exp(\omega^o_{il})}."
            ),
            (
                r"\mathcal L=-\sum_{b,t}\log p_\theta" + "\n"
                r"(s_{\mathrm{res}}^{(b,t)}\mid "
                r"s_{\mathrm{res}}^{(b,<t)},s_{\mathrm{prefix}}^{(b)})."
            ),
        ]
        for tex in cases:
            with self.subTest(tex=tex):
                self.assert_fenced(tex, "$$" + tex + "$$")

    def test_parentheses_and_square_brackets_survive(self):
        tex = r"p(w_{n+1:L}\mid X_{1:n})=\prod_{l=n+1}^{L}p(w_l\mid X_{1:n},w_{n+1:l-1})."
        self.assert_fenced(tex)
        self.assert_fenced(r"z=\mathrm{MLP}\!\left([x_{1,1};x_{1,2}]\right).")

    def test_display_angle_brackets_and_text_tokens_are_unchanged(self):
        # Meteor uses literal <tor> inside \texttt. Fenced TeX needs no inline fix.
        tex = (
            r"\mathcal L_R=-\sum_t\log p(R_t\mid I,Q,R_{<t},\texttt{<tor>}),"
            "\n" + r"S<-0.5,\quad n>1."
        )
        result = self.assert_fenced(tex)
        self.assertIn(r"\texttt{<tor>}", result)
        self.assertNotIn(r"\lt", result)
        self.assertNotIn(r"\gt", result)

    def test_legacy_display_uses_math_fence(self):
        tex = r"\left\{x:x>0\right\}"
        self.assert_fenced(tex, "\\[\n" + tex + "\n\\]\n")

    def test_existing_math_fences_remain_idempotent(self):
        for marker in ("```", "~~~", "````"):
            source = marker + "math\n" + r"x_{<t}+\!y_*" + "\n" + marker + "\n"
            with self.subTest(marker=marker):
                normalized, _ = normalize_math_details(source)
                self.assertEqual(normalized, source)
                self.assertEqual(list(math_expressions(normalized))[0]["tex"], r"x_{<t}+\!y_*")

    def test_display_boundaries_do_not_merge_with_surrounding_prose(self):
        tex = r"\{x\}"
        result = self.assert_fenced(tex, "前文。\n\n$$" + tex + "$$\n\n后文。\n")
        self.assertRegex(result, r"前文。\n\n+```math\n")
        self.assertRegex(result, r"\n```\n\n+后文。")


class InlineMathRegressionTests(unittest.TestCase):
    def test_angle_brackets_are_replaced_only_inside_inline_math(self):
        cases = [
            (r"y_{<t}", r"y_{\lt t}"),
            (r"n>1", r"n\gt 1"),
            (r"x<y", r"x\lt y"),
            (r"S<-0.5", r"S\lt -0.5"),
            (r"\pi_S(y^+)>\pi_T(y^+)", r"\pi_S(y^+)\gt \pi_T(y^+)"),
            (r"a<b<c", r"a\lt b\lt c"),
        ]
        for original, expected in cases:
            for syntax in ("$" + original + "$", "$`" + original + "`$", r"\(" + original + r"\)"):
                with self.subTest(original=original, syntax=syntax):
                    normalized, _ = normalize_math_details(syntax)
                    self.assertEqual(normalized, "$`" + expected + "`$")
                    self.assertEqual(list(math_expressions(normalized))[0]["tex"], expected)
                    self.assertEqual(normalize_math_details(normalized)[0], normalized)

    def test_observed_double_html_encoding_no_longer_changes_inline_tex(self):
        # These are the actual DeepSeek-VL2 / InternVL1.5 failure shapes.
        self.assertEqual(observed_github_inline_text(r"y_{<t}"), r"y_{&lt;t}")
        self.assertEqual(observed_github_inline_text(r"n>1"), r"n&gt;1")
        for original in (r"y_{<t}", r"n>1", r"x<y", r"S<-0.5"):
            with self.subTest(original=original):
                normalized, _ = normalize_math_details("$`" + original + "`$")
                tex = list(math_expressions(normalized))[0]["tex"]
                self.assertEqual(observed_github_inline_text(tex), tex)
                self.assertNotRegex(tex, r"[<>]|&(?:amp;)?(?:lt|gt);")

    def test_existing_comparison_macros_are_unchanged(self):
        source = r"$`y_{\lt t},n\gt 1,a\le b,c\ge d`$"
        self.assertEqual(normalize_math_details(source)[0], source)

    def test_lists_and_tables_preserve_inline_comparison_tex(self):
        source = "- 条件 $n>1$。\n\n| 条件 | 说明 |\n| --- | --- |\n| $x<y$ | 下界 |\n"
        result, _ = normalize_math_details(source)
        self.assertEqual(result, source.replace("$n>1$", r"$`n\gt 1`$").replace("$x<y$", r"$`x\lt y`$"))
        self.assertEqual([item["tex"] for item in math_expressions(result)], [r"n\gt 1", r"x\lt y"])

    def test_prose_html_links_and_plain_comparisons_are_untouched(self):
        source = '<details>\n<summary>n>1</summary>\n\n<https://example.org/a?q=1>\n\nx<y。\n</details>\n'
        self.assertEqual(normalize_math_details(source)[0], source)


class CodeProtectionRegressionTests(unittest.TestCase):
    def test_non_math_code_is_byte_for_byte_unchanged(self):
        samples = [
            "```python\nvalue = '$x<y$'\n# \\[ x \\]\n```\n",
            "~~~text\n$$\na_*+b_*\n$$\n~~~\n",
            "````markdown\n```math\nx<y\n```\n$$z$$\n````\n",
            "代码 `$$x<y$$` 和 ``$`n>1`$``。\n",
            "多行代码 `first line\n$x<y$\nlast line`。\n",
            "正文。\n\n    code = '$x<y$'\n    \\operatorname{f}(x)\n",
            "```python\n'$x<y$'\n",  # An unclosed ordinary code fence.
        ]
        for source in samples:
            with self.subTest(source=source):
                self.assertEqual(normalize_math_details(source)[0], source)
                self.assertEqual(list(math_expressions(source)), [])

    def test_mixed_math_and_code_keep_expression_order_and_code_payload(self):
        source = "条件 $n>1$。\n\n```python\nraw = '$x<y$'\n```\n\n$$a_*+b_*$$\n"
        normalized, _ = normalize_math_details(source)
        self.assertIn("```python\nraw = '$x<y$'\n```", normalized)
        self.assertEqual(
            [(item["tex"], item["display"]) for item in math_expressions(normalized)],
            [(r"n\gt 1", False), (r"a_*+b_*", True)],
        )
        self.assertEqual(normalize_math_details(normalized)[0], normalized)


if __name__ == "__main__":
    unittest.main()
