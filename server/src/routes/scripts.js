const express = require('express');
const router = express.Router();
const multer = require('multer');
const { PDFParse } = require('pdf-parse');
const { analyzeScript } = require('../services/geminiService');

// Configure multer for in-memory file storage
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB max
    fileFilter: (req, file, cb) => {
        const allowed = [
            'application/pdf',
            'text/plain',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ];
        if (allowed.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error('Only PDF, TXT, and DOC/DOCX files are supported.'));
        }
    }
});

/**
 * POST /api/scripts/analyze
 * Upload a script file and receive AI-driven analysis
 */
router.post('/analyze', upload.single('script'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded. Please upload a script file.' });
        }

        console.log(`📜 Received script: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)} KB, ${req.file.mimetype})`);

        // ─── Extract text from the file ───
        let scriptText = '';

        if (req.file.mimetype === 'text/plain') {
            scriptText = req.file.buffer.toString('utf-8');
        } else if (req.file.mimetype === 'application/pdf') {
            const parser = new PDFParse({ data: req.file.buffer });
            try {
                const pdfData = await parser.getText();
                scriptText = pdfData.text;
            } finally {
                await parser.destroy();
            }
        } else {
            // For DOC/DOCX, try basic text extraction from buffer
            scriptText = req.file.buffer.toString('utf-8');
        }

        if (!scriptText || scriptText.trim().length < 100) {
            return res.status(400).json({
                error: 'Could not extract enough text from the uploaded file. Please ensure the file contains readable script text (at least 100 characters).'
            });
        }

        console.log(`📝 Extracted ${scriptText.length} characters. Sending to Gemini for analysis...`);

        // ─── Send to Gemini AI ───
        const analysis = await analyzeScript(scriptText);

        console.log(`✅ Analysis complete for: ${req.file.originalname}`);

        res.json({
            success: true,
            filename: req.file.originalname,
            fileSize: req.file.size,
            textLength: scriptText.length,
            analysis
        });

    } catch (error) {
        console.error('Script analysis error:', error);

        if (error.message?.includes('SAFETY')) {
            return res.status(400).json({
                error: 'The script content was flagged by AI safety filters. Please try a different script.'
            });
        }
        if (error.message?.includes('quota') || error.message?.includes('429')) {
            return res.status(429).json({
                error: 'API rate limit reached. Please wait a moment and try again.'
            });
        }

        res.status(500).json({
            error: error.message || 'Failed to analyze script. Please try again.'
        });
    }
});

module.exports = router;

