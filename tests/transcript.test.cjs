const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('service/frontend/web_service.html', 'utf8');
const code = html.slice(html.indexOf('const SPEAKER_COLORS'), html.indexOf('async function renameSpeaker'));
function render(meeting) {
  const box = { style: {} }; const hint = { style: {} };
  const context = { document: { getElementById: id => id === 'transcript-text' ? box : hint },
    esc: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') };
  vm.createContext(context); vm.runInContext(code, context);
  context.renderTranscript(meeting); return {box, hint};
}
test('all inline scripts parse', () => {
  for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)) new vm.Script(match[1]);
});
test('same speaker retains individual ranges and requested column order', () => {
  const meeting = {segments: [{speaker:'SPEAKER_00',start:0,end:5,text:'first'}, {speaker:'SPEAKER_00',start:65,end:3661,text:'second'}]};
  const before = JSON.stringify(meeting); const {box, hint} = render(meeting);
  assert.ok(box.innerHTML.includes('Speaker</th><th scope="col">발화 시간</th><th scope="col">발화 내용'));
  assert.ok(box.innerHTML.includes('00:00 – 00:05')); assert.ok(box.innerHTML.includes('01:05 – 1:01:01'));
  assert.equal((box.innerHTML.match(/<tr>/g)||[]).length, 3);
  assert.equal(hint.style.display, ''); assert.equal(JSON.stringify(meeting), before);
});
test('segments without diarization retain timestamps without rename buttons', () => {
  const {box, hint} = render({segments:[{start:1,end:2,text:'hello'}, {speaker:'UNKNOWN',start:3,end:4,text:'world'}]});
  assert.ok(box.innerHTML.includes('00:01 – 00:02')); assert.ok(box.innerHTML.includes('화자 미상'));
  assert.doesNotMatch(box.innerHTML, /<button/); assert.equal(hint.style.display, 'none');
});
test('legacy text and missing times remain without fabricated timestamps', () => {
  const {box} = render({transcript:'old\nrecord'});
  assert.ok(box.innerHTML.includes('old\nrecord')); assert.ok(box.innerHTML.includes('— – —'));
  assert.ok(render({segments:[{start:null,end:'bad',text:'text'}]}).box.innerHTML.includes('— – —'));
  assert.equal(render({}).box.textContent, '(발화 기록 없음)');
});
test('speaker names, IDs and text are escaped', () => {
  const id = 'speaker" onfocus="alert(1)';
  const {box} = render({segments:[{speaker:id,start:0,end:1,text:'<script>alert(1)</script>'}],speaker_names:{[id]:'<b>name</b>'}});
  assert.ok(box.innerHTML.includes('speaker&quot; onfocus=&quot;alert'));
  assert.ok(box.innerHTML.includes('&lt;b&gt;name&lt;/b&gt;')); assert.doesNotMatch(box.innerHTML, /<script>/);
});
