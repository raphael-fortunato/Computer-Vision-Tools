#!/usr/bin/env python3
import os
import urllib.request
import re
import threading
import posixpath
import urllib.parse
import argparse
import socket
import time
import hashlib
import pickle
import signal
import imghdr

# config
output_dir = './bing'  # default output dir
adult_filter = False  # Do not disable adult filter by default
socket.setdefaulttimeout(2)

tried_urls = []
image_md5s = {}
in_progress = 0
urlopenheader = {
    'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) \
            Gecko/20100101 Firefox/60.0'}

def download(pool_sema: threading.Semaphore, url: str, output_dir: str, purl: str):
    global in_progress

    if url in tried_urls or url is None:
        return
    pool_sema.acquire()
    in_progress += 1
    path = urllib.parse.urlsplit(url).path
    filename = posixpath.basename(path).split(
        '?')[0]  # Strip GET parameters from filename
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

        image_md5s[md5_key] = filename
        imgpath = os.path.join(output_dir, filename)
        imagefile = open(imgpath, 'wb')
        imagefile.write(image)

        file = open(os.path.join(output_dir, "urls",
                                 txt_name), "w")
        file.write(purl)
        file.close()

        print(purl)
        imagefile.close()
        print("OK: " + filename)
        tried_urls.append(url)
    except Exception as e:
        print("FAIL: " + filename)
    finally:
        pool_sema.release()
        in_progress -= 1


def fetch_images_from_keyword(pool_sema: threading.Semaphore, keyword: str, output_dir: str, filters: str, limit: int):
    current = 0
    last = ''
    while True:
        time.sleep(0.1)
        if in_progress > 10:
            continue
        request_url = 'https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(keyword) + '&first=' + str(
            current) + '&count=35&adlt=' + adlt + '&qft=' + ('' if filters is None else filters)
        request = urllib.request.Request(
            request_url, None, headers=urlopenheader)
        response = urllib.request.urlopen(request)
        html = response.read().decode('utf8')
        links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
        purls = re.findall('purl&quot;:&quot;(.*?)&quot;', html)
        try:
            if links[-1] == last:
                return
            for index, (link, purl) in enumerate(zip(links,purls)):
                if limit is not None and current + index >= limit:
                    return
                t = threading.Thread(target=download, args=(
                    pool_sema, link, output_dir, purl))
                t.start()
                current += 1
            last = links[-1]
        except IndexError:
            print('No search results for "{0}"'.format(keyword))
            return


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
    if adult_filter:
        adlt = ''
    else:
        adlt = 'off'
    if args.adult_filter_off:
        adlt = 'off'
    elif args.adult_filter_on:
        adlt = ''
    pool_sema = threading.BoundedSemaphore(args.threads)
    fetch_images_from_keyword(
        pool_sema, args.search_string, output_dir, args.filters, args.limit)
