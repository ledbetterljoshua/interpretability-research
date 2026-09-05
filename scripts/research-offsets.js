'use strict';
(async () => {
  const byId = id => document.getElementById(id);
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const read = async path => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Could not load ${path}`);
    return response.json();
  };
  const percent = n => `${(100 * n).toFixed(1)}%`;
  const tableRow = (target, values) => {
    const row = element('tr');
    values.forEach((value, i) => {
      const cell = element(i === 0 ? 'th' : 'td', String(value));
      if (i === 0) cell.scope = 'row';
      row.append(cell);
    });
    target.append(row);
  };
  try {
    const [summary, manifest, responseSummary] = await Promise.all([
      read('data/conditional_routes/summary.json'), read('data/conditional_routes/run.json'), read('data/response_predictor/summary.json')
    ]);
    if (manifest.status !== 'complete') throw new Error('The measurement run is incomplete.');
    const labels = {capital:'Capital demonstration',wrong_answer:'Wrong-answer demonstration',other_attribute:'Language demonstration',neutral:'Neutral sentence',shuffled:'Shuffled words',position_only:'Position only'};
    Object.entries(summary.mechanism.fresh.conditions).forEach(([key, condition]) =>
      tableRow(byId('prefix-results'), [labels[key], ...['full','direct','output_bias','natural'].map(k => condition[k].correct)]));
    const forms = {possessive:'Possessive capital',qa:'Question / answer capital',language:'Primary language'};
    Object.entries(summary.scope).forEach(([key, form]) =>
      tableRow(byId('scope-results'), [forms[key], ...['baseline','capital','neutral','other_attribute'].map(k => `${form.modes[k].correct}/${form.eligible}`)]));
    const responseGroups = [['Old validation lattice', responseSummary.original.fresh],
      ['New possessive subsets', responseSummary.scope.possessive],
      ['New QA subsets', responseSummary.scope.qa], ['New language subsets', responseSummary.scope.language],
      ['New subsets pooled', responseSummary.scope_pooled]];
    responseGroups.forEach(([name, group]) => tableRow(byId('response-results'),
      [name, ...['frozen','last_mlp','all_mlps'].map(k => percent(group[k].correctness_disagreement))]));
    const errors = summary.routes.fresh.means;
    byId('prediction-summary').textContent = `Across the ten validation countries and all 256 subsets per country, the predictor disagrees with actual correct-answer status in ${percent(errors.correctness_disagreement)} of conditions. It predicts a different top token in ${percent(errors.top_disagreement)}. These are exhaustive averages over this intervention set; the 2,560 conditions are not independent examples.`;
    const select = byId('country');
    manifest.cases.forEach(name => {
      const [split,country] = name.split('__');
      const option = element('option', `${country} · ${split === 'fresh' ? 'validation' : 'development'}`);
      option.value = name;
      select.append(option);
    });
    // Arrange visual columns by layer while preserving the experiment's bit order.
    const inputs = [];
    for (const kind of ['attn','mlp']) {
      for (let layer = 8; layer <= 11; layer++) {
        const index = (layer - 8) * 2 + (kind === 'mlp' ? 1 : 0);
        const label = element('label');
        const input = element('input');
        input.type = 'checkbox'; input.checked = true;
        input.addEventListener('change', render);
        inputs[index] = input;
        label.append(input, document.createTextNode(`${kind === 'attn' ? 'Attention' : 'MLP'} ${layer}`));
        byId('switch-grid').append(label);
      }
    }
    let current = null;
    let requestNumber = 0;
    const cache = new Map();
    function render() {
      if (!current) return;
      const mask = inputs.reduce((value, input, i) => value | (input.checked ? 1 << i : 0), 0);
      const observed = current.records[mask];
      const mode = byId('predictor').value;
      const result = {actual:observed.actual, predicted:current.response.records[mask].predictions[mode]};
      const predictionTitle = byId('predictor').selectedOptions[0].textContent;
      const grid = element('div', undefined, 'outcome-grid');
      for (const [key, title] of [['actual','Actual model'],['predicted',predictionTitle]]) {
        const resultMetrics = result[key];
        const card = element('div', undefined, 'outcome-card');
        card.append(element('h3', title), element('p', resultMetrics.top.trim() || '(whitespace token)', 'answer'),
          element('p', `Correct answer: ${current.capital} · rank ${resultMetrics.rank} · probability ${percent(resultMetrics.p)}`));
        grid.append(card);
      }
      const same = result.actual.top_id === result.predicted.top_id;
      const verdict = element('p', `${same ? 'Same top token.' : 'Different top tokens.'} ${(result.actual.rank === 1) === (result.predicted.rank === 1) ? 'Correct-answer status agrees.' : 'Correct-answer status differs.'}`, 'verdict');
      byId('outcome').replaceChildren(grid, verdict);
    }
    async function loadCountry() {
      const request = ++requestNumber;
      const name = select.value;
      current = null;
      byId('outcome').replaceChildren(element('p','Loading this country’s measurements…'));
      try {
        if (!cache.has(name)) {
          const [observed, response] = await Promise.all([read(`data/conditional_routes/${name}.json`), read(`data/response_predictor/${name}.json`)]);
          cache.set(name, {...observed, response});
        }
        if (request !== requestNumber) return;
        current = cache.get(name);
        render();
      } catch (error) {
        if (request === requestNumber) byId('outcome').textContent = `${error.message}. Please reload to retry.`;
      }
    }
    select.addEventListener('change', loadCountry);
    byId('predictor').addEventListener('change', render);
    document.querySelectorAll('[data-mask]').forEach(button => button.addEventListener('click', () => {
      const mask = Number(button.dataset.mask);
      inputs.forEach((input, i) => { input.checked = Boolean((mask >> i) & 1); });
      render();
    }));
    select.disabled = false;
    byId('sites').disabled = false;
    await loadCountry();
  } catch (error) {
    byId('outcome').textContent = `${error.message} Serve this directory with a local HTTP server to load the saved data.`;
  }
})();
