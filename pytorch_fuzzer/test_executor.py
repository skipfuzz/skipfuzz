

import socket
import sys
import traceback
import os
from collections import defaultdict
import time


HOST = '127.0.0.1'  # Standard loopback interface address (localhost)

PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

try:
    PORT = int(sys.argv[1])
except:
    print('failed to parse PORT from', sys.argv)
    PORT = 65432
# https://github.com/nedbat/coveragepy/issues/683
prev_tracefunc = None



print('[script server] will run on port', PORT)

prev_line = 0
prev_filename = ''

coverage_data = defaultdict(set)
#https://github.com/fuzzitdev/pythonfuzz/blob/master/pythonfuzz/tracer.py
# license: APACHE 2.0
def trace(frame, event, arg):

    global prev_tracefunc
    # print('prev', prev_tracefunc)
    prev_tracefunc(frame, event, arg)

    sys.settrace(trace)
    # print('trace start', frame, event, arg)

    filename = frame.f_code.co_filename

    # print('filename?', filename)
    if 'tensorflow' not in filename and 'torch' not in filename:
        return trace

    global prev_line
    global prev_filename

    if event != 'line':
        return trace

    func_filename = frame.f_code.co_filename
    func_line_no = frame.f_lineno
    # print('trace!', func_filename, func_line_no)
    if func_filename != prev_filename:
        coverage_data[func_filename + prev_filename].add((prev_line, func_line_no))
    else:
        coverage_data[func_filename].add((prev_line, func_line_no))

    prev_line = func_line_no
    prev_filename = func_filename

    return trace

def get_coverage():
    return sum(map(len, coverage_data.values()))

# def coverage_to_string():
#     return ','.join('(' + item[0] + ',' + str(item[1]) + ')' for item in coverage_data)



def send_back_coverage(conn):
    conn.sendall('==='.encode('utf-8'))
    # print('sent ', "===")
    conn.sendall(str(get_coverage()).encode('utf-8'))
    # print('sent ', str(get_coverage()).encode('utf-8'))
    conn.sendall('rann'.encode('utf-8'))
    # print('sent rann')

def main():
    global prev_tracefunc
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        print('[script server] starting to bind')
        s.bind((HOST, PORT))
        print('[script server] fin binding,', PORT)

        s.listen()
        print('[script server] s listening')
        conn, addr = s.accept()

        with conn:

            print('Connected by', addr)

            data = ''
            while True:
                s.settimeout(210)

                # print('script-server: receiving', time.time())
                data_raw = conn.recv(2048)
                # print('script-server: received', time.time())
                # print('[server] received')
                data_raw_str = data_raw.decode('utf-8')
                if len(data_raw_str) > 0:
                    # print('[server recv]', data_raw_str)
                    data += data_raw_str
                # print('[server]', data)


                if len(data) == 0:
                    continue
                if not data.endswith('=='):
                    continue
                data = data.split('==')[0]
                if data == 'done':
                    break

                # print('script-server: will read', time.time())
                # print('[script-server] will read', data.strip())
                with open(data.strip()) as file:
                    lines = '\n'.join(file.readlines())


                    # BUT! for freefuzz, store the whole exception, want to perform further analysis on the thrown exceptions
                    # lines = lines.replace('results["err"] = "Error:"+str(e)', 'results["err"] = e')

                data = ''
                try:
                    # clear prev coverage
                    # coverage_data.clear()

                    # prev_tracefunc = sys._getframe(0).f_trace
                    # sys.settrace(trace)

                    # print('[script server] executing...')
                    results = {} # freefuzz assumes the presence of a `results` variable and writes to it
                    # print(lines)

                    # print('script-server: will exec', time.time())
                    exec(lines)

                    # print('script-server: done exec', time.time())
                    # print('[script server] done executing...')
                    # print('fin succ execution')
                    # sys.settrace(None)

                    # sys.settrace(prev_tracefunc)

                    if len(results) > 0: # if `results` was modified, we are running tests from freefuzz
                        if "err" in results:
                            raise results['err']

                except Exception as e:

                    # print('[server]', tb)
                    # print('[server]', type(e))
                    # print('[server]', e)
                    try:
                        return_value =  type(e).__name__ + '. message:' + str(e)
                    except TypeError:
                        tb = traceback.format_exc()
                        return_value = '\t' + tb
                    # print('script-server: done exec with exception', time.time())
                    # print('[server] sent back', return_value.encode('utf-8')[:400])
                    conn.sendall(return_value.encode('utf-8')[:400])
                    send_back_coverage(conn)

                    # print('script-server: send back', time.time())
                    continue
                except BaseException as be:
                    tb = traceback.format_exc()
                    # print('[server]', tb)
                    # print('[server]', be)
                    return_value =  type(be).__name__ + '. message:' + str(be) + '\t' + tb
                    # print('script-server: done exec with exception', time.time())
                    # print('[server] sent back', return_value.encode('utf-8')[:400])
                    conn.sendall(return_value.encode('utf-8')[:400])
                    send_back_coverage(conn)
                    # print('script-server: send back', time.time())
                    continue


                conn.sendall("0".encode('utf-8'))
                send_back_coverage(conn)
                # print('[[script server]] sent back', "0".encode('utf-8'))
                # print('script-server: send back', time.time())

            s.shutdown(socket.SHUT_RDWR)
            s.close()

            print('after the loop')
    print('script server end')
main()
print('[script server] exiting')