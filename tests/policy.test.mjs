import {test} from 'node:test';
import assert from 'node:assert/strict';
import {translationAllowed,accepted} from '../web/policy.mjs';
const real={trained_on:'consented-real-krsl',real_data_verified:true,evaluation:{signer_disjoint:true}};
test('synthetic and unknown models never translate',()=>{
  for(const m of [{},{trained_on:'синтетика',...{real_data_verified:true,evaluation:{signer_disjoint:true}}},{...real,trained_on:'synthetic'}]) {
    assert.equal(translationAllowed(m),false); assert.equal(accepted([[0,.99],[1,.01]],m),false);
  }
});
test('reject ambiguity and low confidence',()=>{
  assert.equal(accepted([[0,.6],[1,.4]],real),false);
  assert.equal(accepted([[0,.71],[1,.69]],real),false);
  assert.equal(accepted([[0,NaN],[1,.1]],real),false);
  assert.equal(accepted([[0,.9],[1,.1]],real),true);
});
