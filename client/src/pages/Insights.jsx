import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import Chart from 'react-apexcharts';
import { movieService, trendsService } from '../services/api';
import './Insights.css';

// ── Helpers ──────────────────────────────────────────
const fmt = (v) => {
    if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
    if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
    if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
    return `$${v}`;
};
const fmtShort = (v) => {
    if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(0)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
    return v;
};

const CATEGORY_COLORS = {
    Blockbuster: '#ffa500', 'Super Hit': '#22c55e', Hit: '#3b82f6',
    Average: '#f97316', 'Below Average': '#ef4444', Flop: '#dc2626',
    Disaster: '#991b1b', Unknown: '#666'
};

// ── Shared ApexCharts dark theme ─────────────────────
const darkTheme = {
    mode: 'dark',
    palette: 'palette1',
    monochrome: { enabled: false },
};
const darkChart = {
    background: 'transparent',
    foreColor: '#666',
    fontFamily: "'Work Sans', sans-serif",
    toolbar: { show: false },
    zoom: { enabled: false },
    animations: {
        enabled: true,
        easing: 'easeinout',
        speed: 800,
        dynamicAnimation: { enabled: true, speed: 400 },
    },
};
const darkGrid = {
    borderColor: 'rgba(255,255,255,0.06)',
    strokeDashArray: 4,
    xaxis: { lines: { show: false } },
};
const darkTooltip = {
    theme: 'dark',
    style: { fontSize: '12px', fontFamily: "'Work Sans', sans-serif" },
    x: { show: true },
};

const Insights = () => {
    const [industry, setIndustry] = useState('');
    const [loading, setLoading] = useState(true);
    const [stats, setStats] = useState(null);
    const [yearlyData, setYearlyData] = useState([]);
    const [genreData, setGenreData] = useState([]);
    const [seasonalData, setSeasonalData] = useState([]);
    const [regionalData, setRegionalData] = useState([]);
    const [scatterData, setScatterData] = useState([]);
    const [topDirectors, setTopDirectors] = useState([]);
    const [topActors, setTopActors] = useState([]);
    const [youtubeData, setYoutubeData] = useState([]);
    const [productionData, setProductionData] = useState([]);
    const [openingData, setOpeningData] = useState([]);
    const [criticData, setCriticData] = useState([]);

    const industries = [
        { value: '', label: 'All Industries' },
        { value: 'hollywood', label: 'Hollywood' },
        { value: 'bollywood', label: 'Bollywood' },
        { value: 'tollywood', label: 'Tollywood' },
        { value: 'kollywood', label: 'Kollywood' },
        { value: 'mollywood', label: 'Mollywood' },
    ];

    useEffect(() => {
        (async () => {
            setLoading(true);
            try {
                const p = industry ? { industry } : {};
                const [st, yr, gn, se, re, sc, di, ac, yt, ph, ow, ca] = await Promise.all([
                    movieService.getStats(industry || undefined),
                    trendsService.getYearly(p),
                    trendsService.getGenres(p),
                    trendsService.getSeasonal(p),
                    trendsService.getRegional(),
                    trendsService.getBudgetRevenue(p),
                    trendsService.getTalent('directors', { ...p, limit: 10 }),
                    trendsService.getTalent('actors', { ...p, limit: 10 }),
                    trendsService.getYoutubeHype(p).catch(() => ({ data: [] })),
                    trendsService.getProductionHouses(p).catch(() => ({ data: [] })),
                    trendsService.getOpeningWeekend(p).catch(() => ({ data: [] })),
                    trendsService.getCriticAudience(p).catch(() => ({ data: [] })),
                ]);
                setStats(st.data);
                setYearlyData(yr.data);
                setGenreData(gn.data);
                setSeasonalData(se.data);
                setRegionalData(re.data);
                setScatterData(sc.data);
                setTopDirectors(di.data);
                setTopActors(ac.data);
                setYoutubeData(yt.data);
                setProductionData(ph.data);
                setOpeningData(ow.data);
                setCriticData(ca.data);
            } catch (e) { console.error('Insights fetch error:', e); }
            finally { setLoading(false); }
        })();
    }, [industry]);

    // ── Animation ────────────────────────────────
    const card = (i) => ({
        hidden: { opacity: 0, y: 40 },
        show: { opacity: 1, y: 0, transition: { delay: i * 0.08, duration: 0.6, ease: [.22, 1, .36, 1] } },
    });

    // ── Chart Configs ────────────────────────────

    // 1 ─ Yearly Trends (combo: bars for count, area for revenue, line for rating)
    const yearlyChart = useMemo(() => {
        const years = yearlyData.map(d => String(d._id));
        const revenueData = yearlyData.map(d => Math.round(d.avgRevenue || 0));
        const countData = yearlyData.map(d => d.totalMovies || 0);
        const ratingData = yearlyData.map(d => parseFloat((d.avgRating || 0).toFixed(1)));

        return {
            options: {
                chart: {
                    ...darkChart,
                    type: 'line',
                    height: 420,
                    stacked: false,
                    zoom: { enabled: true, type: 'x' },
                },
                theme: darkTheme,
                colors: ['#ff4d00', '#ffa500', '#3b82f6'],
                stroke: { curve: 'smooth', width: [0, 3, 3] },
                fill: {
                    type: ['solid', 'gradient', 'solid'],
                    gradient: { shadeIntensity: 1, opacityFrom: 0.45, opacityTo: 0.05, stops: [0, 90, 100] },
                },
                plotOptions: {
                    bar: { borderRadius: 4, columnWidth: '50%' },
                },
                xaxis: {
                    categories: years,
                    labels: {
                        rotate: -45,
                        rotateAlways: years.length > 15,
                        style: { colors: '#888', fontSize: '11px', fontFamily: "'Work Sans', sans-serif" },
                    },
                    axisBorder: { show: true, color: '#2a2a2a' },
                    axisTicks: { show: true, color: '#2a2a2a' },
                    crosshairs: { show: true, stroke: { color: '#ff4d00', width: 1, dashArray: 3 } },
                },
                yaxis: [
                    {
                        seriesName: 'Movies Released',
                        title: { text: 'Movies Released', style: { color: '#ff4d00', fontSize: '12px', fontWeight: 600 } },
                        labels: { formatter: (v) => Math.round(v), style: { colors: '#ff4d00' } },
                        min: 0,
                    },
                    {
                        seriesName: 'Avg Revenue',
                        opposite: true,
                        title: { text: 'Avg Revenue', style: { color: '#ffa500', fontSize: '12px', fontWeight: 600 } },
                        labels: { formatter: fmt, style: { colors: '#ffa500' } },
                        min: 0,
                    },
                    {
                        seriesName: 'Avg Rating',
                        opposite: true,
                        title: { text: 'Rating (/10)', style: { color: '#3b82f6', fontSize: '12px', fontWeight: 600 } },
                        labels: { formatter: (v) => v.toFixed(1), style: { colors: '#3b82f6' } },
                        min: 0,
                        max: 10,
                        tickAmount: 5,
                    },
                ],
                grid: darkGrid,
                tooltip: {
                    ...darkTooltip,
                    shared: true,
                    intersect: false,
                    y: {
                        formatter: (v, { seriesIndex }) => {
                            if (seriesIndex === 0) return `${v} movies`;
                            if (seriesIndex === 1) return fmt(v);
                            return `${v.toFixed(1)} / 10`;
                        },
                    },
                },
                legend: {
                    position: 'top',
                    horizontalAlign: 'right',
                    labels: { colors: '#999' },
                    fontSize: '12px',
                    fontFamily: "'Work Sans', sans-serif",
                    markers: { size: 5, strokeWidth: 0 },
                    itemMargin: { horizontal: 12 },
                },
                markers: {
                    size: [0, 4, 4],
                    strokeWidth: 0,
                    hover: { size: 6 },
                },
                dataLabels: { enabled: false },
                annotations: {
                    xaxis: yearlyData.length > 0 ? [{
                        x: '2020',
                        borderColor: '#ef4444',
                        strokeDashArray: 4,
                        label: {
                            text: 'COVID-19',
                            borderColor: '#ef4444',
                            style: { color: '#fff', background: '#ef4444', fontSize: '10px', fontFamily: "'Work Sans', sans-serif", padding: { left: 6, right: 6, top: 2, bottom: 2 } },
                            position: 'top',
                        },
                    }] : [],
                },
            },
            series: [
                { name: 'Movies Released', type: 'bar', data: countData },
                { name: 'Avg Revenue', type: 'area', data: revenueData },
                { name: 'Avg Rating', type: 'line', data: ratingData },
            ],
        };
    }, [yearlyData]);

    // 2 ─ Success Distribution (radialBar donut)
    const successChart = useMemo(() => {
        const cats = stats?.categoryStats || [];
        const total = cats.reduce((s, c) => s + c.count, 0) || 1;
        const labels = cats.map(c => c._id);
        const values = cats.map(c => Math.round((c.count / total) * 100));
        const colors = cats.map(c => CATEGORY_COLORS[c._id] || '#666');
        return {
            options: {
                chart: { ...darkChart, type: 'donut' },
                theme: darkTheme,
                labels,
                colors,
                plotOptions: {
                    pie: {
                        donut: {
                            size: '62%',
                            labels: {
                                show: true,
                                name: { show: true, color: '#fff', fontSize: '14px', fontFamily: "'Syne', sans-serif" },
                                value: { show: true, color: '#999', fontSize: '24px', fontWeight: 700, fontFamily: "'Syne', sans-serif", formatter: (v) => `${v}%` },
                                total: { show: true, label: 'Total', color: '#666', fontSize: '13px', fontFamily: "'Work Sans', sans-serif", formatter: () => `${total} movies` },
                            },
                        },
                    },
                },
                stroke: { width: 2, colors: ['#0a0a0a'] },
                dataLabels: { enabled: false },
                legend: { position: 'bottom', labels: { colors: '#999' }, fontFamily: "'Work Sans', sans-serif", fontSize: '12px' },
                tooltip: { ...darkTooltip, y: { formatter: (v) => `${v} movies` } },
            },
            series: cats.map(c => c.count),
        };
    }, [stats]);

    // 3 ─ Release Patterns (bar + line combo)
    const seasonalChart = useMemo(() => ({
        options: {
            chart: { ...darkChart, type: 'bar', height: 380, stacked: false },
            theme: darkTheme,
            colors: ['#ff4d00', '#22c55e'],
            plotOptions: { bar: { borderRadius: 6, columnWidth: '55%' } },
            fill: {
                type: ['gradient', 'solid'],
                gradient: { shade: 'dark', type: 'vertical', shadeIntensity: 0.4, opacityFrom: 0.9, opacityTo: 0.6, stops: [0, 100] },
            },
            xaxis: { categories: seasonalData.map(d => d.month), labels: { style: { colors: '#666', fontSize: '12px' } } },
            yaxis: [
                { title: { text: 'Avg Revenue', style: { color: '#666' } }, labels: { formatter: fmt, style: { colors: '#666' } } },
                { opposite: true, title: { text: 'Success Rate', style: { color: '#666' } }, labels: { formatter: v => `${(v * 100).toFixed(0)}%`, style: { colors: '#666' } }, min: 0, max: 1 },
            ],
            grid: darkGrid,
            tooltip: { ...darkTooltip, shared: true, intersect: false, y: { formatter: (v, { seriesIndex }) => seriesIndex === 0 ? fmt(v) : `${(v * 100).toFixed(1)}%` } },
            legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#999' } },
            stroke: { width: [0, 3], curve: 'smooth' },
            dataLabels: { enabled: false },
        },
        series: [
            { name: 'Avg Revenue', type: 'bar', data: seasonalData.map(d => Math.round(d.avgRevenue || 0)) },
            { name: 'Success Rate', type: 'line', data: seasonalData.map(d => parseFloat((d.successRate || 0).toFixed(3))) },
        ],
    }), [seasonalData]);

    // 4 ─ Genre Treemap
    const genreTreemap = useMemo(() => {
        const palette = [
            '#ff4d00', '#3b82f6', '#10b981', '#a855f7', '#f59e0b',
            '#ef4444', '#06b6d4', '#ec4899', '#84cc16', '#f97316',
            '#6366f1', '#14b8a6',
        ];
        const items = genreData.slice(0, 12).map((g, i) => ({
            x: g._id,
            y: g.count,
            fillColor: palette[i % palette.length],
            avgRevenue: g.avgRevenue,
            avgRating: g.avgRating,
            avgROI: g.avgROI,
        }));
        return {
            options: {
                chart: { ...darkChart, type: 'treemap' },
                theme: darkTheme,
                plotOptions: {
                    treemap: {
                        distributed: true,
                        enableShades: false,
                        useFillColorAsStroke: false,
                        borderRadius: 6,
                        colorScale: { ranges: items.map((it, i) => ({ from: i, to: i, color: it.fillColor })) },
                    },
                },
                stroke: { width: 2, colors: ['#0a0a0a'] },
                dataLabels: {
                    enabled: true,
                    style: { fontSize: '14px', fontFamily: "'Syne', sans-serif", fontWeight: 700, colors: ['#fff'] },
                    formatter: (text, op) => {
                        if (op.value < 100) return text;
                        return [text, `${op.value}`];
                    },
                    offsetY: -2,
                },
                tooltip: {
                    custom: ({ dataPointIndex }) => {
                        const g = genreData[dataPointIndex];
                        if (!g) return '';
                        return `<div style="background:#141414;border:1px solid #2a2a2a;border-radius:10px;padding:12px 16px;font-family:'Work Sans',sans-serif;min-width:160px">
                            <div style="color:#fff;font-weight:700;font-size:14px;margin-bottom:8px;font-family:'Syne',sans-serif">${g._id}</div>
                            <div style="color:#ffa500;font-size:12px;margin-bottom:3px">🎬 ${g.count} movies</div>
                            <div style="color:#ff4d00;font-size:12px;margin-bottom:3px">💰 Avg Revenue: ${fmt(g.avgRevenue || 0)}</div>
                            <div style="color:#3b82f6;font-size:12px;margin-bottom:3px">⭐ Avg Rating: ${(g.avgRating || 0).toFixed(1)}/10</div>
                            ${g.avgROI ? `<div style="color:#10b981;font-size:12px">📈 Avg ROI: ${Math.round(g.avgROI)}%</div>` : ''}
                        </div>`;
                    },
                },
                legend: { show: false },
            },
            series: [{ data: items }],
        };
    }, [genreData]);

    // 5 ─ Genre Radar
    const genreRadar = useMemo(() => {
        const top = genreData.slice(0, 8);
        return {
            options: {
                chart: { ...darkChart, type: 'radar', height: 360 },
                theme: darkTheme,
                colors: ['#ff4d00', '#3b82f6', '#a855f7'],
                xaxis: { categories: top.map(g => g._id), labels: { style: { colors: '#999', fontSize: '11px' } } },
                yaxis: { show: false },
                stroke: { width: 2 },
                fill: { opacity: [0.25, 0.15, 0.1] },
                markers: { size: 3 },
                legend: { position: 'bottom', labels: { colors: '#999' }, fontSize: '12px' },
                tooltip: darkTooltip,
                plotOptions: { radar: { polygons: { strokeColors: 'rgba(255,255,255,0.08)', connectorColors: 'rgba(255,255,255,0.08)' } } },
                dataLabels: { enabled: false },
            },
            series: [
                { name: 'Revenue ($M)', data: top.map(g => Math.round((g.avgRevenue || 0) / 1e6)) },
                { name: 'Rating (x10)', data: top.map(g => Math.round((g.avgRating || 0) * 10)) },
                { name: 'ROI Score', data: top.map(g => Math.round(Math.min(Math.max(g.avgROI || 0, 0), 500) / 5)) },
            ],
        };
    }, [genreData]);

    // 6 ─ Industry Comparison
    const industryChart = useMemo(() => ({
        options: {
            chart: { ...darkChart, type: 'bar', height: 340 },
            theme: darkTheme,
            colors: ['#3b82f6', '#ff4d00'],
            plotOptions: { bar: { borderRadius: 6, columnWidth: '60%', grouped: true } },
            xaxis: { categories: regionalData.map(r => r._id?.charAt(0).toUpperCase() + r._id?.slice(1)), labels: { style: { colors: '#666', fontSize: '12px' } } },
            yaxis: { labels: { formatter: fmt, style: { colors: '#666' } } },
            grid: darkGrid,
            tooltip: { ...darkTooltip, y: { formatter: fmt } },
            legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#999' } },
            dataLabels: { enabled: false },
            fill: { type: 'gradient', gradient: { shade: 'dark', type: 'vertical', shadeIntensity: 0.3, opacityFrom: 0.95, opacityTo: 0.7 } },
        },
        series: [
            { name: 'Avg Budget', data: regionalData.map(r => Math.round(r.avgBudget || 0)) },
            { name: 'Avg Revenue', data: regionalData.map(r => Math.round(r.avgRevenue || 0)) },
        ],
    }), [regionalData]);

    // 7 ─ Budget vs Revenue Scatter
    const scatterChart = useMemo(() => {
        const grouped = {};
        scatterData.forEach(d => {
            const cat = d.category || 'Unknown';
            if (!grouped[cat]) grouped[cat] = [];
            grouped[cat].push([d.budget, d.revenue]);
        });
        const colors = Object.keys(grouped).map(k => CATEGORY_COLORS[k] || '#666');
        return {
            options: {
                chart: { ...darkChart, type: 'scatter', height: 400, zoom: { enabled: true, type: 'xy' } },
                theme: darkTheme,
                colors,
                xaxis: { title: { text: 'Budget', style: { color: '#666' } }, labels: { formatter: fmtShort, style: { colors: '#666' } }, tickAmount: 8 },
                yaxis: { title: { text: 'Revenue', style: { color: '#666' } }, labels: { formatter: fmtShort, style: { colors: '#666' } }, tickAmount: 8 },
                grid: darkGrid,
                markers: { size: 5, strokeWidth: 0, hover: { size: 8 } },
                tooltip: {
                    ...darkTooltip,
                    custom: ({ seriesIndex, dataPointIndex, w }) => {
                        const cat = w.config.series[seriesIndex]?.name;
                        const point = w.config.series[seriesIndex]?.data?.[dataPointIndex];
                        if (!point) return '';
                        return `<div style="background:#141414;border:1px solid #2a2a2a;border-radius:10px;padding:10px 14px;font-family:'Work Sans',sans-serif;font-size:12px">
                            <div style="color:#fff;font-weight:600;margin-bottom:4px">${cat}</div>
                            <div style="color:#999">Budget: ${fmt(point[0])}</div>
                            <div style="color:#999">Revenue: ${fmt(point[1])}</div>
                            <div style="color:${CATEGORY_COLORS[cat] || '#666'}">ROI: ${((point[1] - point[0]) / point[0] * 100).toFixed(0)}%</div>
                        </div>`;
                    },
                },
                legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#999' }, fontSize: '11px' },
                dataLabels: { enabled: false },
            },
            series: Object.entries(grouped).map(([name, data]) => ({ name, data })),
        };
    }, [scatterData]);

    // 8 ─ Top Directors (horizontal bar)
    const directorsChart = useMemo(() => ({
        options: {
            chart: { ...darkChart, type: 'bar', height: 380 },
            theme: darkTheme,
            colors: ['#ff4d00'],
            plotOptions: { bar: { horizontal: true, borderRadius: 6, barHeight: '65%', distributed: true } },
            xaxis: { labels: { formatter: fmt, style: { colors: '#666' } } },
            yaxis: { labels: { style: { colors: '#999', fontSize: '12px', fontFamily: "'Work Sans', sans-serif" } } },
            grid: { ...darkGrid, yaxis: { lines: { show: false } }, xaxis: { lines: { show: true } } },
            tooltip: { ...darkTooltip, y: { formatter: fmt } },
            legend: { show: false },
            dataLabels: { enabled: false },
            fill: { type: 'gradient', gradient: { shade: 'dark', type: 'horizontal', shadeIntensity: 0.2, opacityFrom: 1, opacityTo: 0.7 } },
            colors: ['#ff4d00', '#ff6a1a', '#ff8533', '#ffa500', '#ffb833', '#ffc94d', '#ffd666', '#ffe280', '#ffee99', '#fff5b3'],
        },
        series: [{ name: 'Avg Revenue', data: topDirectors.slice(0, 10).map(d => ({ x: d._id || 'Unknown', y: Math.round(d.avgRevenue || 0) })) }],
    }), [topDirectors]);

    // 9 ─ Top Actors (horizontal bar)
    const actorsChart = useMemo(() => ({
        options: {
            chart: { ...darkChart, type: 'bar', height: 380 },
            theme: darkTheme,
            colors: ['#3b82f6'],
            plotOptions: { bar: { horizontal: true, borderRadius: 6, barHeight: '65%', distributed: true } },
            xaxis: { labels: { formatter: fmt, style: { colors: '#666' } } },
            yaxis: { labels: { style: { colors: '#999', fontSize: '12px', fontFamily: "'Work Sans', sans-serif" } } },
            grid: { ...darkGrid, yaxis: { lines: { show: false } }, xaxis: { lines: { show: true } } },
            tooltip: { ...darkTooltip, y: { formatter: fmt } },
            legend: { show: false },
            dataLabels: { enabled: false },
            fill: { type: 'gradient', gradient: { shade: 'dark', type: 'horizontal', shadeIntensity: 0.2, opacityFrom: 1, opacityTo: 0.7 } },
            colors: ['#3b82f6', '#2563eb', '#1d4ed8', '#60a5fa', '#93c5fd', '#2979ff', '#448aff', '#82b1ff', '#5b9cff', '#7ab3ff'],
        },
        series: [{ name: 'Avg Revenue', data: topActors.slice(0, 10).map(a => ({ x: a._id || 'Unknown', y: Math.round(a.avgRevenue || 0) })) }],
    }), [topActors]);

    // 10 ─ YouTube Hype vs Box Office (scatter)
    const youtubeHypeChart = useMemo(() => ({
        options: {
            chart: { ...darkChart, type: 'scatter', height: 400, zoom: { enabled: true, type: 'xy' } },
            theme: darkTheme,
            colors: ['#ff4d00', '#ffa500', '#3b82f6', '#10b981', '#ef4444'],
            xaxis: {
                title: { text: 'Trailer Views', style: { color: '#888', fontSize: '12px' } },
                labels: { formatter: (v) => v >= 1e6 ? (v / 1e6).toFixed(0) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(0) + 'K' : v, style: { colors: '#666' } },
            },
            yaxis: {
                title: { text: 'Box Office Revenue', style: { color: '#888', fontSize: '12px' } },
                labels: { formatter: fmt, style: { colors: '#666' } },
            },
            grid: darkGrid,
            tooltip: {
                custom: ({ seriesIndex, dataPointIndex, w }) => {
                    const point = w.config.series[seriesIndex]?.data?.[dataPointIndex];
                    if (!point) return '';
                    return `<div style="background:#141414;border:1px solid #2a2a2a;border-radius:10px;padding:10px 14px;font-family:'Work Sans',sans-serif">
                        <div style="color:#fff;font-weight:600;font-size:13px;margin-bottom:6px">${point.title || ''}</div>
                        <div style="color:#ffa500;font-size:12px">Views: ${(point.x || 0).toLocaleString()}</div>
                        <div style="color:#ff4d00;font-size:12px">Revenue: ${fmt(point.y || 0)}</div>
                    </div>`;
                },
            },
            markers: { size: 8, strokeWidth: 0, hover: { size: 12 } },
            legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#999' }, fontSize: '11px' },
            dataLabels: { enabled: false },
        },
        series: (() => {
            const cats = {};
            youtubeData.forEach(m => {
                const cat = m.category || 'Unknown';
                if (!cats[cat]) cats[cat] = [];
                cats[cat].push({ x: m.trailerViews, y: m.revenue, title: m.title });
            });
            return Object.entries(cats).map(([name, data]) => ({ name, data }));
        })(),
    }), [youtubeData]);

    // 11 ─ Production House Rankings (horizontal bar)
    const productionChart = useMemo(() => ({
        options: {
            chart: { ...darkChart, type: 'bar', height: 420 },
            theme: darkTheme,
            colors: ['#ff4d00'],
            plotOptions: { bar: { horizontal: true, borderRadius: 6, barHeight: '62%', distributed: true } },
            xaxis: { labels: { formatter: fmt, style: { colors: '#666' } } },
            yaxis: { labels: { style: { colors: '#999', fontSize: '11px', fontFamily: "'Work Sans', sans-serif" } } },
            grid: { ...darkGrid, yaxis: { lines: { show: false } }, xaxis: { lines: { show: true } } },
            tooltip: {
                ...darkTooltip,
                y: { formatter: fmt },
                custom: ({ seriesIndex, dataPointIndex, w }) => {
                    const d = productionData[dataPointIndex];
                    if (!d) return '';
                    return `<div style="background:#141414;border:1px solid #2a2a2a;border-radius:10px;padding:10px 14px;font-family:'Work Sans',sans-serif">
                        <div style="color:#fff;font-weight:600;font-size:13px;margin-bottom:6px">${d._id}</div>
                        <div style="color:#ffa500;font-size:12px">Total Revenue: ${fmt(d.totalRevenue)}</div>
                        <div style="color:#ff4d00;font-size:12px">Avg Revenue: ${fmt(d.avgRevenue)}</div>
                        <div style="color:#888;font-size:12px">${d.movieCount} movies · ⭐ ${(d.avgRating || 0).toFixed(1)}</div>
                    </div>`;
                },
            },
            legend: { show: false },
            dataLabels: { enabled: false },
            fill: { type: 'gradient', gradient: { shade: 'dark', type: 'horizontal', shadeIntensity: 0.2, opacityFrom: 1, opacityTo: 0.7 } },
            colors: ['#ff4d00', '#ff6a1a', '#ff8533', '#ffa500', '#ffb833', '#ffc94d', '#ffd666', '#ffe280', '#ffee99', '#fff5b3', '#e65c00', '#cc5200', '#b34700', '#993d00', '#803300'],
        },
        series: [{ name: 'Total Revenue', data: productionData.slice(0, 15).map(d => ({ x: d._id || 'Unknown', y: Math.round(d.totalRevenue || 0) })) }],
    }), [productionData]);

    // 12 ─ Opening Weekend Strength (combo bar + line)
    const openingWeekendChart = useMemo(() => {
        const sorted = [...openingData].sort((a, b) => b.openingPct - a.openingPct).slice(0, 20);
        const titles = sorted.map(d => d.title?.length > 18 ? d.title.slice(0, 18) + '…' : d.title);
        return {
            options: {
                chart: { ...darkChart, type: 'line', height: 400, stacked: false },
                theme: darkTheme,
                colors: ['#ff4d00', '#3b82f6'],
                stroke: { curve: 'smooth', width: [0, 3] },
                plotOptions: { bar: { borderRadius: 4, columnWidth: '55%' } },
                xaxis: {
                    categories: titles,
                    labels: { rotate: -45, rotateAlways: true, style: { colors: '#888', fontSize: '10px', fontFamily: "'Work Sans', sans-serif" }, trim: true, maxHeight: 80 },
                },
                yaxis: [
                    {
                        seriesName: 'Opening Weekend %',
                        title: { text: 'Opening Weekend %', style: { color: '#ff4d00', fontSize: '12px' } },
                        labels: { formatter: (v) => v.toFixed(0) + '%', style: { colors: '#ff4d00' } },
                        min: 0,
                        max: 100,
                    },
                    {
                        seriesName: 'Total Revenue',
                        opposite: true,
                        title: { text: 'Total Revenue', style: { color: '#3b82f6', fontSize: '12px' } },
                        labels: { formatter: fmt, style: { colors: '#3b82f6' } },
                    },
                ],
                grid: darkGrid,
                tooltip: {
                    ...darkTooltip,
                    shared: true,
                    intersect: false,
                    y: {
                        formatter: (v, { seriesIndex }) => seriesIndex === 0 ? v.toFixed(1) + '%' : fmt(v),
                    },
                },
                legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#999' }, fontSize: '12px' },
                dataLabels: { enabled: false },
            },
            series: [
                { name: 'Opening Weekend %', type: 'bar', data: sorted.map(d => d.openingPct || 0) },
                { name: 'Total Revenue', type: 'line', data: sorted.map(d => d.totalRevenue || 0) },
            ],
        };
    }, [openingData]);

    // 13 ─ Critic vs Audience Gap (diverging horizontal bar)
    const criticAudienceChart = useMemo(() => {
        const sorted = [...criticData].sort((a, b) => b.gap - a.gap);
        const titles = sorted.map(d => d.title?.length > 25 ? d.title.slice(0, 25) + '…' : d.title);
        return {
            options: {
                chart: { ...darkChart, type: 'bar', height: Math.max(400, sorted.length * 28) },
                theme: darkTheme,
                plotOptions: { bar: { horizontal: true, borderRadius: 4, barHeight: '65%' } },
                colors: sorted.map(d => d.gap >= 0 ? '#10b981' : '#ef4444'),
                xaxis: {
                    title: { text: 'Gap (Critics − Audience)', style: { color: '#888', fontSize: '12px' } },
                    labels: { formatter: (v) => (v > 0 ? '+' : '') + v + '%', style: { colors: '#666' } },
                },
                yaxis: {
                    labels: { style: { colors: '#999', fontSize: '11px', fontFamily: "'Work Sans', sans-serif" } },
                },
                grid: { ...darkGrid, yaxis: { lines: { show: false } }, xaxis: { lines: { show: true } } },
                tooltip: {
                    custom: ({ dataPointIndex }) => {
                        const d = sorted[dataPointIndex];
                        if (!d) return '';
                        const gapColor = d.gap >= 0 ? '#10b981' : '#ef4444';
                        const label = d.gap >= 0 ? 'Critics loved it more' : 'Audience loved it more';
                        return `<div style="background:#141414;border:1px solid #2a2a2a;border-radius:10px;padding:10px 14px;font-family:'Work Sans',sans-serif">
                            <div style="color:#fff;font-weight:600;font-size:13px;margin-bottom:6px">${d.title}</div>
                            <div style="color:#ffa500;font-size:12px">🍅 Critics: ${d.criticsScore}%</div>
                            <div style="color:#3b82f6;font-size:12px">🍿 Audience: ${d.audienceScore}%</div>
                            <div style="color:${gapColor};font-size:12px;margin-top:4px;font-weight:600">${d.gap > 0 ? '+' : ''}${d.gap}% · ${label}</div>
                        </div>`;
                    },
                },
                legend: { show: false },
                dataLabels: { enabled: false },
                annotations: {
                    xaxis: [{ x: 0, borderColor: '#666', strokeDashArray: 0, label: { text: 'Balanced', borderColor: '#666', style: { color: '#aaa', background: '#1a1a1a', fontSize: '10px' } } }],
                },
            },
            series: [{ name: 'Gap', data: sorted.map((d, i) => ({ x: titles[i], y: d.gap })) }],
        };
    }, [criticData]);

    // ── Stat cards data ──────────────────────────
    const statCards = [
        { label: 'Total Movies', value: stats?.overview?.totalMovies || 0, format: v => v.toLocaleString(), icon: '🎬' },
        { label: 'Avg Budget', value: stats?.overview?.avgBudget || 0, format: fmt, icon: '💰' },
        { label: 'Avg Revenue', value: stats?.overview?.avgRevenue || 0, format: fmt, icon: '💵' },
        { label: 'Avg Rating', value: stats?.overview?.avgRating || 0, format: v => v.toFixed(1) + '/10', icon: '⭐' },
        { label: 'Total Box Office', value: stats?.overview?.totalRevenue || 0, format: fmt, icon: '🏆' },
    ];

    // ── Render ────────────────────────────────────
    return (
        <div className="insights-page">
            {/* ── Header ── */}
            <section className="insights-header">
                <div className="insights-header-bg"></div>
                <div className="container">
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="insights-header-content">
                        <h1 className="insights-title">
                            Data <span className="text-gradient">Insights</span>
                        </h1>
                        <p className="insights-subtitle">
                            Explore trends, patterns, and analytics across 1,600+ movies from Hollywood and Indian cinema
                        </p>
                        <div className="industry-filter">
                            {industries.map(ind => (
                                <button key={ind.value} type="button" className={`filter-pill ${industry === ind.value ? 'active' : ''}`} onClick={() => setIndustry(ind.value)}>
                                    {ind.label}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* ── Content ── */}
            <section className="insights-content">
                <div className="container">
                    {loading ? (
                        <div className="insights-loading">
                            <div className="loading-spinner"></div>
                            <p>Crunching the numbers...</p>
                        </div>
                    ) : (
                        <div className="bento-grid">
                            {/* ── Row 1: Stat Cards ── */}
                            {statCards.map((s, i) => (
                                <motion.div key={i} className="bento-cell stat-cell" variants={card(i)} initial="hidden" animate="show">
                                    <div className="stat-card">
                                        <span className="stat-icon">{s.icon}</span>
                                        <span className="stat-value">{s.format(s.value)}</span>
                                        <span className="stat-label">{s.label}</span>
                                    </div>
                                </motion.div>
                            ))}

                            {/* ── Row 2: Yearly Trends (large) + Success Donut ── */}
                            <motion.div className="bento-cell span-4" variants={card(5)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                <div className="chart-card">
                                    <div className="chart-card-accent"></div>
                                    <div className="chart-header">
                                        <div className="chart-label-bar"></div>
                                        <div>
                                            <h3 className="chart-title">Year-over-Year Trends</h3>
                                            <p className="chart-desc">Revenue, movie count, and average rating across years</p>
                                        </div>
                                    </div>
                                    <Chart options={yearlyChart.options} series={yearlyChart.series} type="line" height={420} />
                                </div>
                            </motion.div>

                            <motion.div className="bento-cell span-2" variants={card(6)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                <div className="chart-card">
                                    <div className="chart-card-accent"></div>
                                    <div className="chart-header">
                                        <div className="chart-label-bar"></div>
                                        <div>
                                            <h3 className="chart-title">Success Distribution</h3>
                                            <p className="chart-desc">Movie breakdown by success category</p>
                                        </div>
                                    </div>
                                    <Chart options={successChart.options} series={successChart.series} type="donut" height={340} />
                                </div>
                            </motion.div>

                            {/* ── Row 3: Release Patterns (full width) ── */}
                            <motion.div className="bento-cell span-6" variants={card(7)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                <div className="chart-card">
                                    <div className="chart-card-accent"></div>
                                    <div className="chart-header">
                                        <div className="chart-label-bar"></div>
                                        <div>
                                            <h3 className="chart-title">Release Month Patterns</h3>
                                            <p className="chart-desc">When should you release a movie? Revenue & success rate by month</p>
                                        </div>
                                    </div>
                                    <Chart options={seasonalChart.options} series={seasonalChart.series} type="bar" height={380} />
                                </div>
                            </motion.div>

                            {/* ── Row 4: Genre Treemap + Radar + Industry ── */}
                            <motion.div className="bento-cell span-2" variants={card(8)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                <div className="chart-card">
                                    <div className="chart-card-accent"></div>
                                    <div className="chart-header">
                                        <div className="chart-label-bar"></div>
                                        <div>
                                            <h3 className="chart-title">Genre Landscape</h3>
                                            <p className="chart-desc">Block size = number of movies</p>
                                        </div>
                                    </div>
                                    <Chart options={genreTreemap.options} series={genreTreemap.series} type="treemap" height={340} />
                                </div>
                            </motion.div>

                            <motion.div className="bento-cell span-2" variants={card(9)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                <div className="chart-card">
                                    <div className="chart-card-accent"></div>
                                    <div className="chart-header">
                                        <div className="chart-label-bar"></div>
                                        <div>
                                            <h3 className="chart-title">Genre Radar</h3>
                                            <p className="chart-desc">Revenue, rating, and ROI across top genres</p>
                                        </div>
                                    </div>
                                    <Chart options={genreRadar.options} series={genreRadar.series} type="radar" height={340} />
                                </div>
                            </motion.div>

                            {!industry && (
                                <motion.div className="bento-cell span-2" variants={card(10)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                    <div className="chart-card">
                                        <div className="chart-card-accent"></div>
                                        <div className="chart-header">
                                            <div className="chart-label-bar"></div>
                                            <div>
                                                <h3 className="chart-title">Industry Face-Off</h3>
                                                <p className="chart-desc">Budget & revenue by film industry</p>
                                            </div>
                                        </div>
                                        <Chart options={industryChart.options} series={industryChart.series} type="bar" height={340} />
                                    </div>
                                </motion.div>
                            )}

                            {/* ── Row 5: Scatter (full width) ── */}
                            <motion.div className="bento-cell span-6" variants={card(11)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                <div className="chart-card">
                                    <div className="chart-card-accent"></div>
                                    <div className="chart-header">
                                        <div className="chart-label-bar"></div>
                                        <div>
                                            <h3 className="chart-title">Budget vs Revenue</h3>
                                            <p className="chart-desc">Every movie plotted — hover for details, scroll to zoom</p>
                                        </div>
                                    </div>
                                    <Chart options={scatterChart.options} series={scatterChart.series} type="scatter" height={420} />
                                </div>
                            </motion.div>

                            {/* ── Row 6: Directors + Actors ── */}
                            <motion.div className="bento-cell span-3" variants={card(12)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                <div className="chart-card">
                                    <div className="chart-card-accent"></div>
                                    <div className="chart-header">
                                        <div className="chart-label-bar"></div>
                                        <div>
                                            <h3 className="chart-title">Top Directors</h3>
                                            <p className="chart-desc">By average revenue (3+ movies)</p>
                                        </div>
                                    </div>
                                    <Chart options={directorsChart.options} series={directorsChart.series} type="bar" height={380} />
                                </div>
                            </motion.div>

                            <motion.div className="bento-cell span-3" variants={card(13)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                <div className="chart-card">
                                    <div className="chart-card-accent"></div>
                                    <div className="chart-header">
                                        <div className="chart-label-bar"></div>
                                        <div>
                                            <h3 className="chart-title">Top Actors</h3>
                                            <p className="chart-desc">Lead actors by average revenue (3+ movies)</p>
                                        </div>
                                    </div>
                                    <Chart options={actorsChart.options} series={actorsChart.series} type="bar" height={380} />
                                </div>
                            </motion.div>

                            {/* ── Row 7: YouTube Hype + Production Houses ── */}
                            {youtubeData.length > 0 && (
                                <motion.div className="bento-cell span-3" variants={card(14)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                    <div className="chart-card">
                                        <div className="chart-card-accent"></div>
                                        <div className="chart-header">
                                            <div className="chart-label-bar"></div>
                                            <div>
                                                <h3 className="chart-title">YouTube Hype vs Box Office</h3>
                                                <p className="chart-desc">Does trailer buzz predict revenue? Scroll to zoom</p>
                                            </div>
                                        </div>
                                        <Chart options={youtubeHypeChart.options} series={youtubeHypeChart.series} type="scatter" height={400} />
                                    </div>
                                </motion.div>
                            )}

                            {productionData.length > 0 && (
                                <motion.div className={`bento-cell ${youtubeData.length > 0 ? 'span-3' : 'span-6'}`} variants={card(15)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                    <div className="chart-card">
                                        <div className="chart-card-accent"></div>
                                        <div className="chart-header">
                                            <div className="chart-label-bar"></div>
                                            <div>
                                                <h3 className="chart-title">Production House Rankings</h3>
                                                <p className="chart-desc">Top studios by total box office (3+ movies)</p>
                                            </div>
                                        </div>
                                        <Chart options={productionChart.options} series={productionChart.series} type="bar" height={420} />
                                    </div>
                                </motion.div>
                            )}

                            {/* ── Row 8: Opening Weekend Strength (full width) ── */}
                            {openingData.length > 0 && (
                                <motion.div className="bento-cell span-6" variants={card(16)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                    <div className="chart-card">
                                        <div className="chart-card-accent"></div>
                                        <div className="chart-header">
                                            <div className="chart-label-bar"></div>
                                            <div>
                                                <h3 className="chart-title">Opening Weekend Strength</h3>
                                                <p className="chart-desc">What % of total revenue comes from opening weekend?</p>
                                            </div>
                                        </div>
                                        <Chart options={openingWeekendChart.options} series={openingWeekendChart.series} type="line" height={400} />
                                    </div>
                                </motion.div>
                            )}

                            {/* ── Row 9: Critic vs Audience Gap (full width) ── */}
                            {criticData.length > 0 && (
                                <motion.div className="bento-cell span-6" variants={card(17)} initial="hidden" whileInView="show" viewport={{ once: true }}>
                                    <div className="chart-card">
                                        <div className="chart-card-accent"></div>
                                        <div className="chart-header">
                                            <div className="chart-label-bar"></div>
                                            <div>
                                                <h3 className="chart-title">Critic vs Audience Gap</h3>
                                                <p className="chart-desc">🟢 Green = critics loved it more · 🔴 Red = audience loved it more</p>
                                            </div>
                                        </div>
                                        <Chart options={criticAudienceChart.options} series={criticAudienceChart.series} type="bar" height={Math.max(400, criticData.length * 28)} />
                                    </div>
                                </motion.div>
                            )}
                        </div>
                    )}
                </div>
            </section>
        </div>
    );
};

export default Insights;
