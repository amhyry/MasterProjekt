import sys
import locale
import io
import sys
import os
import platform
import math
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

def test0_system_stdout_environment():
    """ Test this with/without stdout, err changing above """

    #string = u'안녕세계'
    #print(string)

    print(sys.getdefaultencoding())
    print(sys.stdout.encoding)



if __name__ == '__main__':
    if sys.stdout.isatty():
        default_encoding = sys.stdout.encoding
    else:
        default_encoding = locale.getpreferredencoding()
    print(default_encoding)
    test0_system_stdout_environment()
    print(os.name)
    print(platform.system())
    x = datetime.now().strftime('%m_%d_%H_%M')
    version = 0
    print(x.split('_'))
    x = sum([int(item) for item in datetime.now().strftime('%m_%d_%H_%M').split('_')])

    print(x)
    x = [int(item) for item in '12_31_23_5'.split('_')]
    print(x)
#reversed(a)
    for i in reversed(x):
        print(len(x)-(x.index(i)+1))
        print(math.pow(10, x.index(i)+1))
        #print(10^(x.index(i)+1))
        #print(i*math.pow(10, x.index(i)+1))
        #l = l + [i * 2]

        #print(i)








    #print(u"some unicode text \N{EURO SIGN}")
    #print(b"some utf-8 encoded bytestring \xe2\x82\xac".decode('utf-8'))
