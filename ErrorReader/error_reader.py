class ErrorReader:
    '''
    Class to read errors from Entic boards
    '''

    def __init__(self, device):
        self.device = device

        self.error_list_pp = ['frequency is greater than the high threshold',
                              'voltage sag on phase A',
                              'voltage sag on phase B',
                              'voltage sag on phase C',
                              'frequency is lesser than the low threshold',
                              'phase loss in Phase A',
                              'phase loss in Phase B',
                              'phase loss in Phase C',
                              'reActive energy total negative',
                              'reActive energy Phase A negative',
                              'reActive energy Phase B negative',
                              'reActive energy Phase C negative',
                              'active energy total negative',
                              'active energy Phase A negative',
                              'active energy Phase B negative',
                              'active energy Phase C negative',
                              'Over current on phase A',
                              'Over current on phase B',
                              'Over current on phase C',
                              'Over voltage on phase A',
                              'Over voltage on phase B',
                              'Over voltage on phase C',
                              'Voltage Phase Sequence Error status',
                              'Current Phase Sequence Error',
                              'Neutral line over current',
                              'All phase sum reactive power no-load condition',
                              'All phase sum active power no-load condition',
                              'All phase arithmetic sum apparent power no-load',
                              'Energy for CF1 Reverse',
                              'Energy for CF2 Reverse',
                              'Energy for CF3 Reverse',
                              'Energy for CF4 Reverse']

        self.error_list_ptz = ['Thermistor Undervoltage 4',
                               'Thermistor Undervoltage 3',
                               'Thermistor Undervoltage 2',
                               'Thermistor Undervoltage 1',
                               'Thermistor Overvoltage 4',
                               'Thermistor Overvoltage 3',
                               'Thermistor Overvoltage 2',
                               'Thermistor Overvoltage 1',
                               'Calculation Error 4',
                               'Calculation Error 3',
                               'Calculation Error 2',
                               'Calculation Error 1',
                               'Pressure Undervoltage 4',
                               'Pressure Undervoltage 3',
                               'Pressure Undervoltage 2',
                               'Pressure Undervoltage 1',
                               'Pressure Overvoltage 4',
                               'Pressure Overvoltage 3',
                               'Pressure Overvoltage 2',
                               'Pressure Overvoltage 1',
                               'RSSI Error',
                               'EEPROM Error']

        self.error_list_flow = ['No received signal',
                                'Bit1 low received signal',
                                'poor received signal',
                                'pipe empty',
                                'hardware failure',
                                'receiving circuits gain in adjusting',
                                'frequency at the frequency output over flow',
                                'current at 4-20mA over flow',
                                'RAM check-sum error',
                                'main clock or timer clock error',
                                'parameters check-sum error',
                                'ROM check-sum error',
                                'temperature circuits error',
                                'reserved',
                                'internal timer over flow',
                                'analog input over range']

    def list_error(self, error_message):
        '''
        Returns list of error based on
        error number
        '''
        error_list = []
        error_msg = []

        if self.device == 1:
            error_msg = '{0:32b}'.format(int(error_message))
            error_list = self.error_list_pp
        elif self.device == 2:
            error_msg = '{0:22b}'.format(int(error_message))
            error_list = self.error_list_ptz
        elif self.device == 3:
            error_msg = '{0:16b}'.format(int(error_message))
            error_list = self.error_list_flow
            # wrote list in reverse by mistake
            error_list.reverse()
        else:
            print("[ ERROR ] Not a supported device")
            return

        err_list = []
        y = 0
        for c in error_msg:
            if c == '1':
                print("[ INFO ] " + str(error_list[y]))
            y += 1
        return err_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="List error message for ptz, pp and flow boards")

    parser.add_argument(
        '--device', '-d', type=int, nargs='?', default='1',
        help="Select '1' for PP, '2' for PTZ, or '3' for flow")

    parser.add_argument(
        '--error', '-e', type=int, nargs='?', default='0',
        help='Error number')

    args = parser.parse_args()

    if args.device < 4 and args.device > 0:
        device_list = ['PP', 'PTZ', 'Flow']

        print('==================================')
        print('ENTIC - Error Reader')
        print("Device: " + str(args.device) + " " + device_list[args.device-1])
        print("Error: " + str(args.error))
        print('==================================')
        print('\tError list')
        err = ErrorReader(args.device)

        err.list_error(args.error)
    else:
        print("[ ERROR ] Not valid device")
