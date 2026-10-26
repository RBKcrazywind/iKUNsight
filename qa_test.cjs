const {chromium}=require('C:/Users/17113/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function main(){
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:900}});
  const errors=[]; page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8766/');
  await page.waitForSelector('#file-input',{state:'attached'});
  await page.locator('#file-input').setInputFiles('D:/学习资料/Project/Project Overview_2026-27.pdf');
  await page.waitForFunction(()=>document.querySelector('#doc-card')?.classList.contains('doc-loaded'),null,{timeout:120000});
  console.log('DOC',await page.locator('#doc-card').innerText());
  await page.locator('#question').fill('这篇资料的主要内容是什么？');
  await page.locator('#ask-form').locator('button').click();
  await page.waitForTimeout(1500);
  console.log('ANSWER', (await page.locator('#conversation').innerText()).slice(-700));
  console.log('EVIDENCE',await page.locator('.evidence-card').count(),'STEPS',await page.locator('.run-step').count(),'ERRORS',errors);
  await page.screenshot({path:'D:/学习资料/Project/scientific-paper-agent/qa.png',fullPage:false});
  await browser.close();
}
main().catch(e=>{console.error(e);process.exit(1)});
