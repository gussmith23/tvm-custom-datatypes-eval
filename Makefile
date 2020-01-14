build-datatypes:
					$(MAKE) -C datatypes/universal-wrapper
					$(MAKE) -C datatypes/bfloat16
					$(MAKE) -C datatypes/nop-type
					$(MAKE) -C datatypes/float32
