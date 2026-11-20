// Builds the running results .docx from content.json (a list of blocks).
// Block types: title, subtitle, h1, h2, h3, p, bullets, numbered, table, image, caption, pagebreak, note.
// Inline **bold** is supported in text. Usage: node build.js content.json out.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, ShadingType, LevelFormat, PageBreak, BorderStyle, Footer, PageNumber, TableOfContents,
} = require("docx");

const [, , contentPath, outPath] = process.argv;
const blocks = JSON.parse(fs.readFileSync(contentPath, "utf8"));
const baseDir = path.dirname(contentPath);
const FONT = "Calibri";
const TEXT_W = 9360; // 6.5in at 1440 dxa/in

function runs(text, opts = {}) {
  const out = [];
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g);
  for (const p of parts) {
    if (!p) continue;
    const bold = p.startsWith("**") && p.endsWith("**");
    out.push(new TextRun({ text: bold ? p.slice(2, -2) : p, bold: bold || opts.bold, italics: opts.italics,
      size: opts.size, color: opts.color, font: FONT }));
  }
  return out;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const children = [];
let listInstance = 0;
for (const b of blocks) {
  switch (b.type) {
    case "title":
      children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: runs(b.text, { size: 44, bold: true }) }));
      break;
    case "subtitle":
      children.push(new Paragraph({ spacing: { after: 200 }, children: runs(b.text, { color: "555555", size: 22 }) }));
      break;
    case "toc":
      children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
      break;
    case "h1":
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, pageBreakBefore: !!b.newPage, children: runs(b.text) }));
      break;
    case "h2":
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: runs(b.text) }));
      break;
    case "h3":
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: runs(b.text) }));
      break;
    case "p":
      children.push(new Paragraph({ spacing: { after: 120 }, children: runs(b.text) }));
      break;
    case "note":
      children.push(new Paragraph({ spacing: { after: 120 }, shading: { type: ShadingType.CLEAR, fill: "F3F1EC" },
        border: { left: { style: BorderStyle.SINGLE, size: 12, color: "2A78D6", space: 6 } },
        children: runs(b.text, { size: 20 }) }));
      break;
    case "bullets":
    case "numbered":
      listInstance += 1;
      for (const item of b.items) {
        const [txt, level] = Array.isArray(item) ? item : [item, 0];
        children.push(new Paragraph({ numbering: { reference: b.type === "bullets" ? "bul" : "num", level, instance: listInstance },
          spacing: { after: 60 }, children: runs(txt) }));
      }
      break;
    case "image": {
      const file = path.resolve(baseDir, b.path);
      const { w, h } = pngSize(file);
      const widthIn = b.widthIn || 6.5;
      const wpx = widthIn * 96, hpx = wpx * h / w;
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
        children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: wpx, height: hpx },
          altText: { title: b.alt || "chart", description: b.alt || "chart", name: path.basename(file) } })] }));
      break;
    }
    case "caption":
      children.push(new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.CENTER,
        children: runs(b.text, { italics: true, size: 18, color: "555555" }) }));
      break;
    case "table": {
      const ncol = b.header.length;
      const widths = b.widths || Array(ncol).fill(Math.floor(TEXT_W / ncol));
      const total = widths.reduce((a, c) => a + c, 0);
      const mk = (cells, header) => new TableRow({ tableHeader: header, children: cells.map((c, i) => new TableCell({
        width: { size: widths[i], type: WidthType.DXA },
        shading: header ? { type: ShadingType.CLEAR, fill: "E8E4DA" } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ alignment: (i > 0 && b.alignRight) ? AlignmentType.RIGHT : AlignmentType.LEFT,
          children: runs(c, { bold: header, size: 19 }) })] })) });
      children.push(new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
        rows: [mk(b.header, true), ...b.rows.map(r => mk(r, false))] }));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      break;
    }
    case "pagebreak":
      children.push(new Paragraph({ children: [new PageBreak()] }));
      break;
    default:
      throw new Error("unknown block " + b.type);
  }
}

const doc = new Document({
  creator: "Anish Srinivasa", title: "Deck content: Phase 1 and Phase 2 results",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "1F2A37", font: FONT }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "1F2A37", font: FONT }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "2F6F6A", font: FONT }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [0, 1].map(l => ({ level: l, format: LevelFormat.BULLET, text: l ? "–" : "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360 + 360 * l, hanging: 260 } } } })) },
    { reference: "num", levels: [0, 1].map(l => ({ level: l, format: l ? LevelFormat.LOWER_LETTER : LevelFormat.DECIMAL,
      text: l ? "%2." : "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360 + 360 * l, hanging: 260 } } } })) },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 18, color: "777777", font: FONT })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(outPath, buf); console.log("wrote", outPath); });
