// Shared, testable decision policy. Softmax confidence is not a calibrated probability.
export function translationAllowed(meta) {
  const source=String(meta?.trained_on||'').toLowerCase();
  return Boolean(source && !/синтет|synthetic|demo/.test(source) && meta.real_data_verified===true && meta.evaluation?.signer_disjoint===true);
}
export function accepted(top, meta) {
  const threshold=meta?.confidence_threshold??0.7;
  return translationAllowed(meta) && Number.isFinite(threshold) && threshold>=0.5 && threshold<=1 && top.length>=2 &&
    top.every(p=>Number.isFinite(p[1])&&p[1]>=0&&p[1]<=1) && top[0][1]>=threshold && top[0][1]-top[1][1]>=0.15;
}
