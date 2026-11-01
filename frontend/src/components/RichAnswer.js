import React from 'react';
import { FileText, Sparkles, CheckCircle2, ChevronRight } from 'lucide-react';

/**
 * Format raw LaTeX math (e.g. `\( \theta_{\text{floor}} = 0.35 \)` or `\( K_0 \)`)
 * into clean, readable typography with unicode symbols.
 */
function formatLatexMath(latex) {
  let s = latex.trim();
  // Remove \text{...}
  s = s.replace(/\\text\{([^}]+)\}/g, '$1');
  // Greek letters
  s = s.replace(/\\theta/g, 'θ');
  s = s.replace(/\\alpha/g, 'α');
  s = s.replace(/\\beta/g, 'β');
  s = s.replace(/\\lambda/g, 'λ');
  s = s.replace(/\\sigma/g, 'σ');
  s = s.replace(/\\delta/g, 'δ');
  // Common subscripts
  s = s.replace(/_0/g, '₀');
  s = s.replace(/_1/g, '₁');
  s = s.replace(/_2/g, '₂');
  s = s.replace(/_k/g, 'ₖ');
  s = s.replace(/_\{([^}]+)\}/g, '_$1');
  // Operators & symbols
  s = s.replace(/\\le/g, '≤');
  s = s.replace(/\\ge/g, '≥');
  s = s.replace(/\\times/g, '×');
  s = s.replace(/\\cdot/g, '·');
  s = s.replace(/\\pm/g, '±');
  s = s.replace(/\\neq/g, '≠');
  s = s.replace(/\\in/g, '∈');
  return s;
}

/**
 * Inline text renderer:
 * - `(Excerpt from File: ...)` -> document badge
 * - `\( ... \)` or `\[ ... \]` -> styled math pill
 * - `**bold**` -> <strong>
 * - `` `code` `` -> <code>
 * - `*italic*` -> <em>
 */
function renderInlineText(text) {
  if (!text) return null;

  // Patterns:
  // 1. (Excerpt from File: ...)
  // 2. \( ... \) or \[ ... \] LaTeX math
  // 3. **bold**
  // 4. `code`
  // 5. *italic*
  const regex = /(\(Excerpt from File:\s*([^)]+)\)|\\\(|\s*\\\[\s*(.*?)\s*\\\)|\s*\\\]|\*\*(.+?)\*\*|`([^`]+)`|\*([^*]+)\*)/g;
  const elements = [];
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      elements.push(text.substring(lastIndex, match.index));
    }

    if (match[2]) {
      // Excerpt from file
      const filename = match[2].trim();
      elements.push(
        <span
          key={`file-${match.index}`}
          className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded-full bg-violet-50 dark:bg-violet-900/30 text-violet-600 dark:text-violet-300 border border-violet-200/60 dark:border-violet-800/40 ml-1.5 align-middle"
          title={`Source: ${filename}`}
        >
          <FileText className="w-3 h-3 flex-shrink-0" />
          <span className="truncate max-w-[180px]">{filename}</span>
        </span>
      );
    } else if (match[3] !== undefined) {
      // LaTeX math: \( ... \) or \[ ... \]
      const formattedMath = formatLatexMath(match[3]);
      elements.push(
        <span
          key={`math-${match.index}`}
          className="inline-flex items-center font-mono text-[11.5px] font-semibold px-1.5 py-0.2 mx-0.5 rounded bg-cyan-50 dark:bg-cyan-950/40 text-cyan-700 dark:text-cyan-300 border border-cyan-200/50 dark:border-cyan-800/50 align-baseline"
        >
          {formattedMath}
        </span>
      );
    } else if (match[4]) {
      // Bold
      elements.push(
        <strong key={`bold-${match.index}`} className="font-semibold text-slate-900 dark:text-white">
          {match[4]}
        </strong>
      );
    } else if (match[5]) {
      // Code
      elements.push(
        <code
          key={`code-${match.index}`}
          className="px-1.5 py-0.5 mx-0.5 rounded bg-slate-100 dark:bg-slate-800 text-cyan-600 dark:text-cyan-400 font-mono text-xs border border-slate-200/60 dark:border-slate-700/60"
        >
          {match[5]}
        </code>
      );
    } else if (match[6]) {
      // Italic
      elements.push(
        <em key={`italic-${match.index}`} className="italic text-slate-700 dark:text-slate-300">
          {match[6]}
        </em>
      );
    }

    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    elements.push(text.substring(lastIndex));
  }

  return elements.length === 1 && typeof elements[0] === 'string' ? elements[0] : elements;
}

/**
 * RichAnswer Component
 * Parses and displays AI explanations, research summaries, and structured outputs
 * with headings, clean lists, styled math, and highlight cards.
 */
export default function RichAnswer({ content }) {
  if (!content) return null;

  // Pre-process raw lines: merge isolated bullet characters with the next line
  const rawLines = content.split('\n');
  const normalizedLines = [];

  for (let i = 0; i < rawLines.length; i++) {
    const trimmed = rawLines[i].trim();
    if (!trimmed) continue;

    // If this line is just an isolated bullet symbol (• or - or *):
    if ((trimmed === '•' || trimmed === '-' || trimmed === '*') && i + 1 < rawLines.length) {
      i++;
      const nextLine = rawLines[i].trim();
      normalizedLines.push(`• ${nextLine}`);
    } else {
      normalizedLines.push(trimmed);
    }
  }

  const blocks = [];
  let currentList = null;

  const flushList = () => {
    if (currentList) {
      blocks.push(currentList);
      currentList = null;
    }
  };

  for (let i = 0; i < normalizedLines.length; i++) {
    const line = normalizedLines[i];

    // ── Headings of any depth: #, ##, ###, ####, #####, ###### ───────────────
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      flushList();
      const level = headingMatch[1].length;
      const text = headingMatch[2].trim();
      const isConclusion = /conclusion|summary|key takeaway|limitations/i.test(text);

      if (level <= 2) {
        blocks.push({ type: 'h2', content: text });
      } else if (level === 3) {
        blocks.push({ type: isConclusion ? 'h3_conclusion' : 'h3', content: text });
      } else {
        // level 4, 5, 6
        blocks.push({ type: 'h4', content: text });
      }
      continue;
    }

    // ── Major Section Headers (e.g. "1. Objective", "2. Key Concepts and Methodology") ───
    // If a numbered line is short (< 60 chars) and has no colon or period in the title, it's a section header
    const majorHeaderMatch = line.match(/^(\d+)\.\s+([A-Z][a-zA-Z0-9\s,&/\-()]{2,60})$/);
    if (majorHeaderMatch && !line.includes(':') && !line.includes('.')) {
      flushList();
      blocks.push({
        type: 'major_section',
        num: majorHeaderMatch[1],
        content: majorHeaderMatch[2].trim(),
      });
      continue;
    }

    // ── Numbered List Item (e.g. "1. **Title**: description") ─────────────────
    const numberedMatch = line.match(/^(\d+)\.\s+(.*)$/);
    if (numberedMatch) {
      if (!currentList || currentList.type !== 'numbered') {
        flushList();
        currentList = { type: 'numbered', items: [] };
      }
      currentList.items.push({
        num: numberedMatch[1],
        content: numberedMatch[2],
      });
      continue;
    }

    // ── Bullet List Item (e.g. "• description", "- description") ──────────────
    const bulletMatch = line.match(/^[-*•]\s+(.*)$/);
    if (bulletMatch) {
      if (!currentList || currentList.type !== 'bullet') {
        flushList();
        currentList = { type: 'bullet', items: [] };
      }
      currentList.items.push(bulletMatch[1]);
      continue;
    }

    // ── Blockquote (e.g. "> note") ───────────────────────────────────────────
    if (line.startsWith('>')) {
      flushList();
      blocks.push({
        type: 'blockquote',
        content: line.replace(/^>\s*/, ''),
      });
      continue;
    }

    // ── Normal Paragraph / Intro ─────────────────────────────────────────────
    flushList();
    // Check if it's an introductory preamble
    const isIntro = i === 0 && /^(to explain|based on the|here is|let's break down)/i.test(line);
    blocks.push({
      type: isIntro ? 'intro' : 'p',
      content: line,
    });
  }

  flushList();

  return (
    <div className="space-y-3.5 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
      {blocks.map((block, idx) => {
        switch (block.type) {
          case 'intro':
            return (
              <div
                key={idx}
                className="p-3 rounded-xl bg-cyan-50/60 dark:bg-cyan-950/20 border border-cyan-200/50 dark:border-cyan-800/40 text-xs text-slate-600 dark:text-slate-300 font-medium flex items-center gap-2"
              >
                <Sparkles className="w-4 h-4 text-cyan-500 flex-shrink-0" />
                <span>{renderInlineText(block.content)}</span>
              </div>
            );

          case 'major_section':
            return (
              <div key={idx} className="pt-4 pb-1">
                <div className="flex items-center gap-2.5 p-2 rounded-xl bg-slate-100/70 dark:bg-slate-800/60 border border-slate-200/60 dark:border-slate-700/60">
                  <span className="w-6 h-6 rounded-lg bg-cyan-500 text-white font-bold text-xs flex items-center justify-center shadow-sm">
                    {block.num}
                  </span>
                  <h2 className="text-sm font-bold text-slate-900 dark:text-white tracking-tight">
                    {renderInlineText(block.content)}
                  </h2>
                </div>
              </div>
            );

          case 'h2':
            return (
              <div key={idx} className="pt-3 pb-1 border-b border-slate-200/60 dark:border-slate-800/60">
                <h2 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cyan-500 flex-shrink-0" />
                  <span>{renderInlineText(block.content)}</span>
                </h2>
              </div>
            );

          case 'h3':
            return (
              <div key={idx} className="pt-2">
                <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2 pl-2.5 border-l-2 border-cyan-500">
                  <span>{renderInlineText(block.content)}</span>
                </h3>
              </div>
            );

          case 'h3_conclusion':
            return (
              <div key={idx} className="pt-3">
                <h3 className="text-sm font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-2 pl-2.5 border-l-2 border-emerald-500">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                  <span>{renderInlineText(block.content)}</span>
                </h3>
              </div>
            );

          case 'h4':
            return (
              <div key={idx} className="pt-2 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-cyan-700 dark:text-cyan-400">
                <ChevronRight className="w-3.5 h-3.5 text-cyan-500 flex-shrink-0" />
                <span>{renderInlineText(block.content)}</span>
              </div>
            );

          case 'numbered':
            return (
              <div key={idx} className="space-y-2 pl-1 my-2">
                {block.items.map((item, itemIdx) => (
                  <div
                    key={itemIdx}
                    className="flex items-start gap-2.5 p-2 rounded-lg bg-slate-50/70 dark:bg-slate-800/40 border border-slate-200/50 dark:border-slate-800/50"
                  >
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-cyan-100 dark:bg-cyan-900/40 text-cyan-700 dark:text-cyan-300 text-xs font-bold flex items-center justify-center mt-0.5">
                      {item.num}
                    </span>
                    <div className="flex-1 text-slate-700 dark:text-slate-200 text-sm">
                      {renderInlineText(item.content)}
                    </div>
                  </div>
                ))}
              </div>
            );

          case 'bullet':
            return (
              <ul key={idx} className="space-y-2 pl-2 my-2">
                {block.items.map((item, itemIdx) => (
                  <li key={itemIdx} className="flex items-start gap-2 text-slate-700 dark:text-slate-300 text-sm">
                    <span className="text-cyan-500 font-bold mt-1 text-xs">•</span>
                    <div className="flex-1 leading-relaxed">
                      {renderInlineText(item)}
                    </div>
                  </li>
                ))}
              </ul>
            );

          case 'blockquote':
            return (
              <blockquote
                key={idx}
                className="pl-3.5 py-1.5 border-l-2 border-violet-400 dark:border-violet-600 italic bg-violet-50/40 dark:bg-violet-950/20 text-slate-600 dark:text-slate-300 text-xs rounded-r-lg my-2"
              >
                {renderInlineText(block.content)}
              </blockquote>
            );

          case 'p':
          default:
            return (
              <p key={idx} className="leading-relaxed text-slate-700 dark:text-slate-200">
                {renderInlineText(block.content)}
              </p>
            );
        }
      })}
    </div>
  );
}
