
def write_output_format_file(_output_format_file, _output_format_dict):
    _fid = open(_output_format_file, 'w')
    # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            if len(_value) == 5:
                # plus dist-info
                _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind)+1, int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), float(_value[4])))
            elif len(_value) == 4:
                # Write Cartesian format output. Since baseline does not estimate track count we use a fixed value.
                _fid.write('{},{},{},{},{},{}\n'.format(int(_frame_ind)+1, int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3])))
            else:
                # Write Polar format output.
                _fid.write('{},{},{},{},{},{}\n'.format(int(_frame_ind)+1, int(_value[0]), 0, int(_value[1]), int(_value[2])))
    _fid.close()