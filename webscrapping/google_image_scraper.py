from selenium import webdriver
import time
import requests
import shutil
import os
import argparse
import urllib
from selenium.webdriver.common.by import By
import posixpath
import imghdr
import hashlib
import threading
import socket
import pickle
import signal

urlopenheader = {
    'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}
tried_urls = []
image_md5s = {}

def save_img(url, directory, link):
    if url in tried_urls or url is None:
        return
    path = urllib.parse.urlsplit(url).path
    filename = posixpath.basename(path).split('?')[0]
    name, ext = os.path.splitext(filename)
    name = name[:36].strip()
    filename = name + ext
    txt_name = name + '.txt'
    try:
        request = urllib.request.Request(url, None, urlopenheader)
        image = urllib.request.urlopen(request).read()
        if not imghdr.what(None, image):
            print('Invalid image, not saving ' + filename)
            return

        md5_key = hashlib.md5(image).hexdigest()
        if md5_key in image_md5s:
            print('Image is a duplicate of ' +
                  image_md5s[md5_key] + ', not saving ' + filename)
            return

        i = 0
        while os.path.exists(os.path.join(output_dir, filename)):
            if hashlib.md5(open(os.path.join(output_dir, filename), 'rb').read()).hexdigest() == md5_key:
                print('Already downloaded ' + filename + ', not saving')
                return
            i += 1
            filename = "%s-%d%s" % (name, i, ext)
            txt_name = "%s-%d%s" % (name, i, '.txt')

        file = open(os.path.join(output_dir, "urls",
                                 txt_name), "w")
        file.write(link)
        file.close()
        imgpath = os.path.join(output_dir, filename)
        image_md5s[md5_key] = filename
        imagefile = open(imgpath, 'wb')
        imagefile.write(image)

        imagefile.close()
        print("OK: " + filename)
        tried_urls.append(url)
    except Exception as e:
        print(e)
        print("FAIL: " + filename)


def find_urls(driver, keywords ,directory):
    keywords = keywords.replace("&", '%26').replace(" ", '+')
    url="https://www.google.co.in/search?q="+keywords+ \
            "&source=lnms&tbm=isch"
    driver.get(url)
    images, purls = [], []
    for _ in range(1000):
        driver.execute_script("window.scrollBy(0,1080)")
        try:
            driver.find_element_by_css_selector('.mye4qd').click()
        except:
            continue
    for  imgurl in driver.find_elements_by_xpath('//img[contains(@class,"rg_i Q4LuWd")]'):
        try:
            imgurl.click()
            time.sleep(.5)
            img = driver.find_element(
                By.XPATH, "/html/body/div[2]/c-wiz/div[4]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img")
            link = driver.find_element(
                    By.XPATH, '//a[contains(@class, "eHAdSb")]')
            images.append(img.get_attribute('src'))
            purls.append(link.get_attribute('href'))
        except Exception as e:
            print(e)
    if len(purls) != len(images):
        print(f"WARNING: Purls {len(purls)}, does not equal Links {len(images)}!")
        import pdb; pdb.set_trace()
    for i, p in zip(images, purls):
        save_img(i, output_dir, p)
    driver.close()


def backup_history(*args):
    download_history = open(os.path.join(
        output_dir, 'download_history.pickle'), 'wb')
    pickle.dump(tried_urls, download_history)
    # We are working with the copy, because length of input variable for pickle must not be changed during dumping
    copied_image_md5s = dict(image_md5s)
    pickle.dump(copied_image_md5s, download_history)
    download_history.close()
    print('history_dumped')
    if args:
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bing image bulk downloader')
    parser.add_argument('-s', '--search-string',
                        help='Keyword to search', required=False)
    parser.add_argument('-f', '--search-file',
                        help='Path to a file containing search strings line by line', required=False)
    parser.add_argument(
        '-o', '--output', help='Output directory', required=False)
    parser.add_argument('--adult-filter-on', help='Enable adult filter',
                        action='store_true', required=False)
    parser.add_argument('--adult-filter-off', help='Disable adult filter',
                        action='store_true', required=False)
    parser.add_argument(
        '--filters', help='Any query based filters you want to append when searching for images, e.g. +filterui:license-L1', required=False)
    parser.add_argument(
        '--limit', help='Make sure not to search for more than specified amount of images.', required=False, type=int)
    parser.add_argument(
        '--threads', help='Number of threads', type=int, default=25)
    args = parser.parse_args()
    if (not args.search_string) and (not args.search_file):
        parser.error(
            'Provide Either search string or path to file containing search strings')
    if args.output:
        output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "urls")):
        os.makedirs(os.path.join(output_dir, "urls"))
    output_dir_origin = output_dir
    signal.signal(signal.SIGINT, backup_history)
    try:
        download_history = open(os.path.join(
            output_dir, 'download_history.pickle'), 'rb')
        tried_urls = pickle.load(download_history)
        image_md5s = pickle.load(download_history)
        download_history.close()
    except (OSError, IOError):
        tried_urls = []
    driver = webdriver.Firefox()
    find_urls(driver, args.search_string, output_dir)
