from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    def log_console(msg):
        print(f"CONSOLE [{msg.type}]: {msg.text}")
        
    page.on("console", log_console)
    page.on("pageerror", lambda exc: print(f"PAGE ERROR: {exc}"))
    
    print("Navigating to http://127.0.0.1:8000...")
    response = page.goto("http://127.0.0.1:8000", wait_until="networkidle")
    print(f"Status: {response.status}")
    
    time.sleep(2)
    twin_val = page.evaluate("window.__twin !== undefined")
    print(f"window.__twin is defined: {twin_val}")
    
    time.sleep(8)
    content = page.content()
    if "FAILED TO LOAD" in content:
        print("FAILED TO LOAD message is present!")
    else:
        print("FAILED TO LOAD message is NOT present!")
        
    browser.close()
