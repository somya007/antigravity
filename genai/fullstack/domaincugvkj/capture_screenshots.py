"""
Capture screenshots of all application outputs using Selenium.
"""
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

OUTPUT_DIR = r"c:\RISK_ASSESSMENT\genai\fullstack\screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--window-size=1200,900")
chrome_options.add_argument("--force-device-scale-factor=1")
chrome_options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 10)

try:
    # ===== SET A Q1: Clock Angle Calculator =====
    print("Capturing SET A Q1 - Clock Angle Calculator...")
    file_path = r"c:\RISK_ASSESSMENT\genai\fullstack\setA_q1_clock_angle\output.html"
    driver.get(f"file:///{file_path}")
    time.sleep(1)
    driver.save_screenshot(os.path.join(OUTPUT_DIR, "setA_q1_output.png"))
    print("  -> setA_q1_output.png saved")

    # ===== SET B Q1: Kids Library =====
    print("Capturing SET B Q1 - Kids Library...")
    file_path = r"c:\RISK_ASSESSMENT\genai\fullstack\setB_q1_kids_library\output.html"
    driver.get(f"file:///{file_path}")
    time.sleep(1)
    driver.save_screenshot(os.path.join(OUTPUT_DIR, "setB_q1_output.png"))
    print("  -> setB_q1_output.png saved")

    # ===== CRM App (SET A Q2 / SET B Q2) =====
    # Screenshot 1: Login Page
    print("Capturing CRM - Login Page...")
    driver.get("http://localhost:3000/login")
    time.sleep(1)
    driver.save_screenshot(os.path.join(OUTPUT_DIR, "crm_login.png"))
    print("  -> crm_login.png saved")

    # Screenshot 2: Login with credentials and capture Home Page
    print("Capturing CRM - Home Page...")
    username_field = driver.find_element(By.ID, "username")
    password_field = driver.find_element(By.ID, "password")
    username_field.send_keys("admin")
    password_field.send_keys("password123")
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
    time.sleep(1)
    driver.save_screenshot(os.path.join(OUTPUT_DIR, "crm_home.png"))
    print("  -> crm_home.png saved")

    # Screenshot 3: Surprise Page
    print("Capturing CRM - Surprise Page...")
    driver.get("http://localhost:3000/surprise")
    time.sleep(1)
    driver.save_screenshot(os.path.join(OUTPUT_DIR, "crm_surprise.png"))
    print("  -> crm_surprise.png saved")

    # Screenshot 4: Customer Page (add a customer first)
    print("Capturing CRM - Customer Page...")
    driver.get("http://localhost:3000/customer?age=25&mobile=9876543210")
    time.sleep(1)
    # Add another customer
    driver.get("http://localhost:3000/customer?age=30&mobile=8765432109")
    time.sleep(1)
    driver.save_screenshot(os.path.join(OUTPUT_DIR, "crm_customer.png"))
    print("  -> crm_customer.png saved")

    print("\nAll screenshots captured successfully!")
    print(f"Screenshots saved in: {OUTPUT_DIR}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    driver.quit()
