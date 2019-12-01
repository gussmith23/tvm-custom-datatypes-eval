import tvm

def test_register():
    tvm.datatype.register('mytype', 132)
