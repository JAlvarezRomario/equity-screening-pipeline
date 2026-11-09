// ==============================================================================
// File: main.js
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
    const root = document.documentElement;
    const body = document.body;
    const store = {
        get: key => { try { return localStorage.getItem(key); } catch (e) { return null; } },
        set: (key, value) => { try { localStorage.setItem(key, value); } catch (e) { /* storage blocked */ } },
    };

    // --- Theme: match browser preference or explicit user choice ---
    const isDark = () => {
        if (root.classList.contains('dark')) return true;
        if (root.classList.contains('light')) return false;
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    };

    const plots = () => Array.from(document.querySelectorAll('.js-plotly-plot')).filter(el => el.offsetParent !== null);

    function themeCharts() {
        if (!window.Plotly) return;
        const dark = isDark();
        const text = dark ? '#c9ced8' : '#344054';
        const grid = dark ? 'rgba(255,255,255,0.08)' : 'rgba(16,24,40,0.08)';
        const line = dark ? 'rgba(255,255,255,0.18)' : 'rgba(16,24,40,0.18)';
        plots().forEach(el => {
            if (!el._fullLayout) return;
            try {
                const update = {'font.color': text, 'legend.font.color': text, 'title.font.color': text};
                Object.keys(el.layout || {}).filter(k => /^[xy]axis\d*$/.test(k)).forEach(k => {
                    update[`${k}.gridcolor`] = grid;
                    update[`${k}.zerolinecolor`] = grid;
                    update[`${k}.linecolor`] = line;
                    update[`${k}.title.font.color`] = text;
                });
                Plotly.relayout(el, update);
            } catch (e) { /* Plot settling */ }
        });
    }

    function applyTheme(theme) {
        if (theme === 'dark') {
            root.classList.add('dark');
            root.classList.remove('light');
        } else if (theme === 'light') {
            root.classList.add('light');
            root.classList.remove('dark');
        } else {
            root.classList.remove('dark', 'light');
        }
        themeCharts();
    }

    // Listen for browser / OS light and dark setting preference changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
            const saved = store.get('theme');
            if (!saved) {
                applyTheme(e.matches ? 'dark' : 'light');
            }
        });
    }

    const themeButton = document.getElementById('theme-toggle');
    if (themeButton) {
        themeButton.addEventListener('click', () => {
            const next = isDark() ? 'light' : 'dark';
            store.set('theme', next);
            applyTheme(next);
        });
    }
    if (isDark()) {
        setTimeout(themeCharts, 50);
    }

    // --- Keep Plotly charts sized to their container ---
    // CSS grid tracks settle after scripts run; relayout with autosize ensures proper render without clipping.
    function resizeCharts() {
        if (!window.Plotly) return;
        plots().forEach(el => {
            if (el._fullLayout) {
                try {
                    Plotly.relayout(el, {autosize: true});
                } catch (e) { /* settling */ }
            }
        });
    }

    window.addEventListener('load', resizeCharts);
    window.addEventListener('resize', resizeCharts);
    setTimeout(resizeCharts, 50);

    if (window.ResizeObserver) {
        const ro = new ResizeObserver(() => resizeCharts());
        document.querySelectorAll('.company-chart, .chart-card, .dist-card').forEach(c => ro.observe(c));
    }

    // --- Tooltips: one floating element for every [data-help] (table headers, company facts) ---
    const tip = document.getElementById('tip');
    function showTip(el) {
        if (!tip || !el.dataset.help) return;
        tip.textContent = el.dataset.help;
        tip.hidden = false;
        const box = el.getBoundingClientRect();
        const width = tip.offsetWidth, height = tip.offsetHeight;
        const left = Math.min(Math.max(8, box.left + box.width / 2 - width / 2), window.innerWidth - width - 8);
        const below = box.bottom + 8 + height < window.innerHeight;
        tip.style.left = `${left}px`;
        tip.style.top = `${below ? box.bottom + 8 : box.top - height - 8}px`;
    }
    const hideTip = () => { if (tip) tip.hidden = true; };
    document.querySelectorAll('[data-help]').forEach(el => {
        if (el.tagName === 'TH') el.tabIndex = 0;
        el.addEventListener('mouseenter', () => showTip(el));
        el.addEventListener('focus', () => showTip(el));
        el.addEventListener('mouseleave', hideTip);
        el.addEventListener('blur', hideTip);
    });
    window.addEventListener('scroll', hideTip, true);

    // --- Company page: business summary clamped to three lines, "Read more" only when it is cut ---
    const summary = document.getElementById('summary');
    const summaryToggle = document.getElementById('summary-toggle');
    if (summary && summaryToggle && summary.scrollHeight > summary.clientHeight + 2) {
        summaryToggle.hidden = false;
        summaryToggle.addEventListener('click', () => {
            const open = summary.classList.toggle('clamp') === false;
            summaryToggle.textContent = open ? 'Show less' : 'Read more';
            summaryToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
    }

    // --- Company page: ← / → open the previous / next company in the ranking (not while typing) ---
    document.addEventListener('keydown', ev => {
        if (ev.altKey || ev.ctrlKey || ev.metaKey || ev.shiftKey) return;
        if (ev.target.closest('input, textarea, select, [contenteditable], [role="tab"]')) return;
        const link = document.querySelector(`.stepper a[data-key="${ev.key}"]`);
        if (link) window.location.href = link.href;
    });

    // --- Highlights: one tab strip over the lists (all lists show without JavaScript) ---
    const tabs = document.querySelector('.tabs');
    if (tabs) {
        const card = tabs.closest('.card');
        const buttons = Array.from(tabs.querySelectorAll('[role="tab"]'));
        const select = id => {
            buttons.forEach(b => {
                const on = b.dataset.panel === id;
                b.setAttribute('aria-selected', on ? 'true' : 'false');
                b.tabIndex = on ? 0 : -1;
                const panel = document.getElementById(b.dataset.panel);
                if (panel) {
                    panel.hidden = !on;
                    if (on && window.Plotly) {
                        panel.querySelectorAll('.js-plotly-plot').forEach(el => {
                            if (el._fullLayout) {
                                try {
                                    Plotly.relayout(el, {autosize: true});
                                } catch (e) { /* settling */ }
                            }
                        });
                    }
                }
            });
            store.set('fq-tab-2', id);
        };
        buttons.forEach((b, i) => {
            b.addEventListener('click', () => select(b.dataset.panel));
            b.addEventListener('keydown', ev => {
                const step = ev.key === 'ArrowRight' ? 1 : ev.key === 'ArrowLeft' ? -1 : 0;
                if (!step) return;
                const next = buttons[(i + step + buttons.length) % buttons.length];
                select(next.dataset.panel);
                next.focus();
            });
        });
        const saved = store.get('fq-tab-2');
        card.classList.add('tabbed');
        tabs.hidden = false;
        select(buttons.some(b => b.dataset.panel === saved) ? saved : buttons[0].dataset.panel);
    }

    // --- Holdings: one switch for the whole page ---
    // On: the combined ranking shows only the portfolio's companies, the other lists are hidden (style.css),
    // the charts dim everything else, and the distribution histograms superimpose the holdings in orange.
    const holdingsButton = document.getElementById('holdings-toggle');
    const heldSlugs = new Set(window.FQ_HELD || []);
    const holdingsOn = () => body.classList.contains('holdings-mode');
    // "No dividend tax": only rows from zero-withholding countries (data-wht0), in the highlight lists (CSS) and the
    // full ranking (filterTable).
    const whtButton = document.getElementById('wht-toggle');
    const whtOn = () => body.classList.contains('wht0-mode');

    function updateHistograms(on) {
        if (!window.Plotly) return;
        const histPlots = Array.from(document.querySelectorAll('.dist-card .js-plotly-plot'));
        histPlots.forEach(el => {
            if (!el.data) return;
            const heldIdx = el.data.findIndex(t => t.name === 'Holdings');
            if (heldIdx < 0) return;
            const runIdx = el.data.findIndex(t => t.name === 'This run');
            const hasBench = el.data.some(t => t.name && t.name !== 'This run' && t.name !== 'Holdings');

            Plotly.restyle(el, {visible: on ? true : false}, [heldIdx]);

            if (!hasBench && runIdx >= 0) {
                const runTrace = el.data[runIdx];
                if (runTrace.type === 'histogram') {
                    Plotly.restyle(el, {histnorm: on ? 'percent' : ''}, [runIdx]);
                    Plotly.relayout(el, {
                        'yaxis.title.text': on ? '% of companies' : 'Count',
                        showlegend: on,
                    });
                } else if (runTrace.type === 'bar' && runTrace.customdata) {
                    const newY = runTrace.customdata.map(row => row[1]);
                    Plotly.restyle(el, {y: [newY]}, [runIdx]);
                    Plotly.relayout(el, {
                        'yaxis.title.text': '% of companies',
                        showlegend: on,
                    });
                }
            }

            if (el.layout && (el.layout.shapes || el.layout.annotations)) {
                const layoutUpdate = {};
                if (el.layout.shapes) {
                    layoutUpdate.shapes = el.layout.shapes.map(s => {
                        if (s.line && (s.line.color === '#f97316' || s.line.dash === 'dashdot')) {
                            return Object.assign({}, s, {visible: on});
                        }
                        return s;
                    });
                }
                if (el.layout.annotations) {
                    layoutUpdate.annotations = el.layout.annotations.map(a => {
                        if (a.text && a.text.includes('Holdings')) {
                            return Object.assign({}, a, {visible: on});
                        }
                        return a;
                    });
                }
                Plotly.relayout(el, layoutUpdate);
            }
        });
    }

    function applyHoldings() {
        const on = holdingsOn();
        if (holdingsButton) holdingsButton.setAttribute('aria-pressed', on ? 'true' : 'false');
        filterTable();
        updateHistograms(on);
        document.dispatchEvent(new Event('fq-holdings'));  // the quadrant listens
    }

    if (holdingsButton) {
        holdingsButton.addEventListener('click', () => {
            body.classList.toggle('holdings-mode');
            applyHoldings();
            const ranking = document.getElementById('ranking');
            if (holdingsOn() && ranking) ranking.scrollIntoView({behavior: 'smooth'});
        });
    }

    // --- Table filtering ---
    // One search over ticker, company, country and industry (cells 1-4); the Holdings switch and quick-filter pills
    // combine to filter the table.
    const search = document.getElementById('ranking-search');
    const count = document.getElementById('ranking-count');
    const rankingTableBody = document.querySelector('#ranking-table tbody');
    const qfButtons = Array.from(document.querySelectorAll('.qf-btn'));
    const showAll = document.getElementById('ranking-show-all');

    // The ranking's first rows are in the table; the rest wait (#ranking-rest: inline text, or a separate script named
    // by data-src that sets window.FQ_RANKING_REST) until a search, filter, sort or "Show all" needs every company.
    // Resolves true once every row is in the table.
    let restLoading = null;
    const restPending = () => rankingTableBody && document.getElementById('ranking-rest') !== null;
    function ensureAllRows() {
        const rest = document.getElementById('ranking-rest');
        if (!rest || !rankingTableBody) return Promise.resolve(true);
        if (restLoading) return restLoading;
        const insert = html => {
            rankingTableBody.insertAdjacentHTML('beforeend', html);
            rest.remove();
            if (showAll) showAll.parentElement.remove();
            return true;
        };
        if (!rest.dataset.src) return Promise.resolve(insert(rest.textContent));
        if (count) count.textContent = 'Loading every company…';
        restLoading = new Promise(resolve => {
            const script = document.createElement('script');
            script.src = rest.dataset.src;
            script.onload = () => {
                const html = window.FQ_RANKING_REST;
                delete window.FQ_RANKING_REST;
                resolve(typeof html === 'string' ? insert(html) : false);
            };
            script.onerror = () => resolve(false);
            document.head.appendChild(script);
        }).then(ok => {
            if (!ok) {
                restLoading = null;
                if (count) count.textContent = 'Could not load every company; try again.';
            } else {
                filterTable();  // the count line, whatever asked for the rows
            }
            return ok;
        });
        return restLoading;
    }

    function filterTable() {
        if (!rankingTableBody) return;
        const query = search ? search.value.trim().toLowerCase() : '';
        const heldOnly = holdingsOn();
        const activeQf = qfButtons.filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.dataset.qf);
        if ((query || heldOnly || activeQf.length || whtOn()) && restPending()) {
            ensureAllRows().then(ok => { if (ok) filterTable(); });
            return;
        }
        const rows = rankingTableBody.querySelectorAll('tr');
        let shown = 0;
        rows.forEach(row => {
            const text = Array.from(row.cells).slice(1, 5).map(c => c.textContent).join(' ').toLowerCase();
            const qfOk = activeQf.every(k => row.dataset[k] === '1');
            const visible = text.includes(query) && (!heldOnly || row.dataset.held === '1') && qfOk
                && (!whtOn() || row.dataset.wht0 === '1');
            row.style.display = visible ? '' : 'none';
            if (visible) shown++;
        });
        const table = rankingTableBody.closest('table');
        const total = Number(table && table.dataset.total) || rows.length;
        if (count) count.textContent = `${shown.toLocaleString()} of ${total.toLocaleString()} companies shown`;
    }

    if (showAll) showAll.addEventListener('click', () => { ensureAllRows().then(ok => { if (ok) filterTable(); }); });

    if (whtButton) {
        whtButton.addEventListener('click', () => {
            body.classList.toggle('wht0-mode');
            whtButton.setAttribute('aria-pressed', whtOn() ? 'true' : 'false');
            filterTable();
        });
    }

    qfButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const on = btn.getAttribute('aria-pressed') === 'true';
            btn.setAttribute('aria-pressed', on ? 'false' : 'true');
            filterTable();
        });
    });

    if (search) search.addEventListener('input', filterTable);
    filterTable();

    // --- Table sorting ---
    // The first click sorts best first (data-first: a criterion where lower is better starts ascending); the next
    // click reverses. Missing values ("–") always go last. Enter or Space sorts from the keyboard.
    const number = text => {
        const clean = text.replace(/[,%+]|\s*(pp|x)$/g, '').trim();  // a whole number, not "0019.HK"
        const n = clean === '' ? NaN : Number(clean);
        return Number.isNaN(n) ? null : n;
    };

    function sortTable(table, column, direction) {
        if (table.id === 'ranking-table' && restPending()) {
            ensureAllRows().then(ok => { if (ok) sortTable(table, column, direction); });
            return;
        }
        const tbody = table.querySelector('tbody');
        const sign = direction === 'ascending' ? 1 : -1;
        const rows = Array.from(tbody.querySelectorAll('tr')).map(row => {
            const text = row.cells[column] ? row.cells[column].textContent.trim() : '';
            return {row, text, value: number(text)};
        });
        rows.sort((a, b) => {
            if (a.value !== null && b.value !== null) return sign * (a.value - b.value);
            if (a.value !== null) return -1;
            if (b.value !== null) return 1;
            const blankA = a.text === '' || a.text === '–', blankB = b.text === '' || b.text === '–';
            if (blankA !== blankB) return blankA ? 1 : -1;
            return sign * a.text.localeCompare(b.text, undefined, {numeric: true});
        });
        tbody.append(...rows.map(r => r.row));
    }

    document.querySelectorAll('table.sortable').forEach(table => {
        const headers = Array.from(table.querySelectorAll('thead th'));
        headers.forEach((header, index) => {
            header.tabIndex = 0;
            const sort = () => {
                const current = header.getAttribute('aria-sort');
                const first = header.dataset.first === 'asc' ? 'ascending'
                    : header.dataset.first === 'desc' ? 'descending' : 'ascending';
                const direction = current ? (current === 'ascending' ? 'descending' : 'ascending') : first;
                headers.forEach(h => h.removeAttribute('aria-sort'));
                header.setAttribute('aria-sort', direction);
                sortTable(table, index, direction);
            };
            header.addEventListener('click', sort);
            header.addEventListener('keydown', ev => {
                if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); sort(); }
            });
        });
    });
});
