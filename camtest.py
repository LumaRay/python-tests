import usb.core
import usb.util

# находим наше устройство
dev = usb.core.find(idVendor=0x04B4, idProduct=0xF9F9, bDeviceClass=0xEF)

# оно было найдено?
if dev is None:
    raise ValueError('Device not found')

c = 1
for config in dev:
    print('config', c)
    print('Interfaces', config.bNumInterfaces)
    '''for i in range(config.bNumInterfaces):
        if dev.is_kernel_driver_active(i):
            dev.detach_kernel_driver(i)
        print(i)'''

    # find_descriptor: что это?
    intf = usb.util.find_descriptor(
        config
    )
    c+=1

# поставим активную конфигурацию. Без аргументов, первая же
# конфигурация будет активной
# dev.set_configuration()

# получим экземпляр источника
# cfg = dev.get_active_configuration()
# intf = cfg[(0,0)]

'''ep = usb.util.find_descriptor(
    0,#intf,
    # сопоставим первый источник данных
    custom_match = \
    lambda e: \
        usb.util.endpoint_direction(e.bEndpointAddress) == \
        usb.util.ENDPOINT_OUT)'''

#assert ep is not None

dev.ctrl_transfer(0x21,0x01,0x0800,0x0600,[0x50,0xff])

# записываем данные
#ep.write('test')