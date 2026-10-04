"""Load the real Three.js module and exercise the decorative settlement."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright


def test_settlement_initializes_and_walkers_move(tmp_path):
    web = Path(__file__).parents[1] / 'src/dashboard/web'

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(web)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=['--enable-unsafe-swiftshader'])
            try:
                page = browser.new_page(viewport={'width': 1440, 'height': 900})
                page.add_init_script("localStorage.setItem('aquaflow-visual-quality', 'PERFORMANCE')")
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('console', lambda message: errors.append(message.text)
                        if 'Renderer initialization failed' in message.text else None)
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='domcontentloaded')
                page.wait_for_function('window.__twinDiagnostics !== undefined', timeout=90000)
                before = page.evaluate('window.__twinDiagnostics.settlement')
                assert before['houses'] == 4
                assert len(before['walkers']) == 4
                assert before['instances'] == {'torso': 4, 'head': 4, 'arm': 8, 'leg': 8}
                page.wait_for_function('(before) => window.__twinDiagnostics.settlement.walkers'
                                       '.some((p, i) => Math.abs(p[2]-before[i][2]) > .1)',
                                       arg=before['walkers'], timeout=30000)
                assert not page.evaluate('window.__twinDiagnostics.rendering.contextLost')
                page.locator('[data-cam="downstream"]').dispatch_event('click')
                page.wait_for_function('Math.abs(window.__twinDiagnostics.target[2]-555) < 1', timeout=30000)
                page.screenshot(path=str(tmp_path / 'downstream_settlement.png'))
                assert not errors, errors
                print(f'SETTLEMENT_SCREENSHOT={tmp_path / "downstream_settlement.png"}')
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
