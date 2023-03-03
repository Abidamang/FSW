																			
									
from acceptanceRunner import AcceptanceSuite, ParameterlessCommandPacket, OutputLevel								
        #Module / Pckage : Acceptance Runer									
		#Imports Method/Functions AcceptanceSuite,, parameterLessCommandPacket, OutputLevel									
		# Dsecription: This test suite tests various AUGSRS requirements. Testing here involves enabling TLM 								There is a class EnableTelemetry() - Passing AcceptanceSuite									
		# transmission, collecting TLM, and verifying we receive expected data																	
																			
class EnableTelemetry(AcceptanceSuite):																	
																			
		# function test_receive_tlm tests the following requirements: 																	
		# AUGSRS-56: The FSW shall collect, package, and transmit telemetry while the FSW is running. 																	
		def test_receive_tlm(self) -> None:																	
		cfs = self.get_transponder('native')																	
																			
		# call cfs to get a command factory																	
		#cmd = cfs.command(cfs.CommandCodes.TO_OUTPUT_ENABLE_CC)																	
		cmd = cfs.get_command_builder(cfs.CommandCodes.TO_OUTPUT_ENABLE_CC)																	
		# set parameters																	
		cmd.set('dest_IP', 'localhost')																	
																			
		cmd.send()																	
		self.wait(5)																	
		self.assertTrue(cfs.count_sample(cfs.Telemetry.CFE_ES_HK_TLM) > 0)																	
																			
		# toggle between SILENT and NORMAL if you would like to supress or see the output from print statments																	
		self._source.output_level = OutputLevel.NORMAL																	
																			
		first_es_hk = cfs.decom_sample(cfs.Telemetry.CFE_ES_HK_TLM, 0)																	
		self.print('initial TLM Timestamp in seconds is', first_es_hk['timestamp']['seconds'], sep=': ')																	
		self.assertTrue(1000000 <= first_es_hk['timestamp']['seconds'])																	
																			
																			
																			
		# function test_check_sw_ver_tlm tests the following requirements: 																	
		# AUGSRS-53: The FSW shall report the active version of the FSW in telemetry.																	
		# AUGSRS-54 The FSW shall include timestamps with telemetry.																	
		# AUGSRS-56: The FSW shall collect, package, and transmit telemetry while the FSW is running. 																	
		# AUGSRS-58: The FSW shall transmit recorded telemetry upon ground command.																	
		def test_check_sw_ver_tlm(self) -> None:																	
		cfs = self.get_transponder('native')																	
																			
		# call cfs to get a command factory																	
		#cmd = cfs.command(cfs.CommandCodes.TO_OUTPUT_ENABLE_CC)																	
		cmd = cfs.get_command_builder(cfs.CommandCodes.TO_OUTPUT_ENABLE_CC)																	
		# set parameters																	
		cmd.set('dest_IP', 'localhost')																	
																			
		# send the command to enable TLM transmits																	
		cmd.send()																	
		self.wait(10)																	
																			
		# verify the FSW sent TLM in response to the enable TLM command sent																	
		self.assertTrue(cfs.count_sample(cfs.Telemetry.CFE_ES_HK_TLM) > 2)																	
																			
		first_es_hk = cfs.decom_sample(cfs.Telemetry.CFE_ES_HK_TLM, 0)																	
		self.print("1st sample Timestamp seconds is", first_es_hk['timestamp']['seconds'], sep=': ')																	
		self.print("CFE Major Ver is", first_es_hk['Payload']['CFEMajorVersion'], sep=': ')																	
		self.print("CFE Minor Ver is", first_es_hk['Payload']['CFEMinorVersion'], sep=': ')																	
		self.print("OSAL Major Ver is", first_es_hk['Payload']['OSALMajorVersion'], sep=': ')																	
		self.print("OSAL Minor Ver is", first_es_hk['Payload']['OSALMinorVersion'], sep=': ')																	
		self.print("PSP Major Ver is", first_es_hk['Payload']['PSPMajorVersion'], sep=': ')																	
		self.print("PSP Minor Ver is", first_es_hk['Payload']['PSPMinorVersion'], sep=': ')																	
		self.print("Boot source is", first_es_hk['Payload']['BootSource'], sep=': ')																	
		self.print("Processor Resets is", first_es_hk['Payload']['ProcessorResets'], sep=': ')																	
		self.print("Perf data start is", first_es_hk['Payload']['PerfDataStart'], sep=': ')																	
		self.print("Perf Data End is", first_es_hk['Payload']['PerfDataEnd'], sep=': ')																	
		second_es_hk = cfs.decom_sample(cfs.Telemetry.CFE_ES_HK_TLM, 1)																	
		self.print("2nd sample Timestamp seconds is", second_es_hk['timestamp']['seconds'], sep=': ')																	
		self.print("CFE Major Ver is", second_es_hk['Payload']['CFEMajorVersion'], sep=': ')																	
		self.print("CFE Minor Ver is", second_es_hk['Payload']['CFEMinorVersion'], sep=': ')																	
		self.print("OSAL Major Ver is", second_es_hk['Payload']['OSALMajorVersion'], sep=': ')																	
		self.print("OSAL Minor Ver is", second_es_hk['Payload']['OSALMinorVersion'], sep=': ')																	
		self.print("PSP Major Ver is", second_es_hk['Payload']['PSPMajorVersion'], sep=': ')																	
		self.print("PSP Minor Ver is", second_es_hk['Payload']['PSPMinorVersion'], sep=': ')																	
		self.print("Boot source is", second_es_hk['Payload']['BootSource'], sep=': ')																	
		self.print("Processor Resets is", second_es_hk['Payload']['ProcessorResets'], sep=': ')																	
		self.print("Perf data start is", second_es_hk['Payload']['PerfDataStart'], sep=': ')																	
		self.print("Perf Data End is", second_es_hk['Payload']['PerfDataEnd'], sep=': ')																	
		third_es_hk = cfs.decom_sample(cfs.Telemetry.CFE_ES_HK_TLM, 2)																	
		self.print("3rd sample Timestamp seconds is", third_es_hk['timestamp']['seconds'], sep=': ')																	
		self.print("CFE Major Ver is", third_es_hk['Payload']['CFEMajorVersion'], sep=': ')																	
		self.print("CFE Minor Ver is", third_es_hk['Payload']['CFEMinorVersion'], sep=': ')																	
		self.print("OSAL Major Ver is", third_es_hk['Payload']['OSALMajorVersion'], sep=': ')																	
		self.print("OSAL Minor Ver is", third_es_hk['Payload']['OSALMinorVersion'], sep=': ')																	
		self.print("PSP Major Ver is", third_es_hk['Payload']['PSPMajorVersion'], sep=': ')																	
		self.print("PSP Minor Ver is", third_es_hk['Payload']['PSPMinorVersion'], sep=': ')																	
		self.print("Boot source is", third_es_hk['Payload']['BootSource'], sep=': ')																	
		self.print("Processor Resets is", third_es_hk['Payload']['ProcessorResets'], sep=': ')																	
		self.print("Perf data start is", third_es_hk['Payload']['PerfDataStart'], sep=': ')																	
		self.print("Perf Data End is", third_es_hk['Payload']['PerfDataEnd'], sep=': ')																	
																			
		# Verify that time stamp is increasing for each TLM received																	
		time_increased = ((first_es_hk['timestamp']['seconds'] < second_es_hk['timestamp']['seconds']) or 																	
		((first_es_hk['timestamp']['seconds'] == second_es_hk['timestamp']['seconds']) and 																	
		(first_es_hk['timestamp']['sub_sec'] < second_es_hk['timestamp']['sub_sec'])))																	
		self.assertTrue(time_increased)																	
																			
		time_increased = ((second_es_hk['timestamp']['seconds'] < third_es_hk['timestamp']['seconds']) or 																	
		((second_es_hk['timestamp']['seconds'] == third_es_hk['timestamp']['seconds']) and 																	
		(second_es_hk['timestamp']['sub_sec'] < third_es_hk['timestamp']['sub_sec'])))																	
		self.assertTrue(time_increased)																	
		# Verify that active FSW version data is consistent in TLM received, 																	
		# expecting versions not to change as time progress																	
		self.assertEqual(first_es_hk['Payload']['CFEMajorVersion'], second_es_hk['Payload']['CFEMajorVersion'])																	
		self.assertEqual(first_es_hk['Payload']['CFEMajorVersion'], third_es_hk['Payload']['CFEMajorVersion'])																	
		self.assertEqual(first_es_hk['Payload']['OSALMinorVersion'], second_es_hk['Payload']['OSALMinorVersion'])																	
		self.assertEqual(first_es_hk['Payload']['OSALMinorVersion'], third_es_hk['Payload']['OSALMinorVersion'])																	
																			
		self.assertEqual(first_es_hk['Payload']['PSPMajorVersion'], second_es_hk['Payload']['PSPMajorVersion'])																	
		self.assertEqual(first_es_hk['Payload']['PSPMajorVersion'], third_es_hk['Payload']['PSPMajorVersion'])																	
																			
																			
		# function test_check_evr_tlm tests the following requirements: 																	
		# AUGSRS-54 The FSW shall include timestamps with telemetry.																	
		# AUGSRS-56: The FSW shall collect, package, and transmit telemetry while the FSW is running. 																	
		# AUGSRS-58: The FSW shall transmit recorded telemetry upon ground command.																	
		# AUGSRS-81: The FSW shall receive, interpret, and execute direct commands from a ground station.																	
		def test_check_evr_tlm(self) -> None:																	
		cfs = self.get_transponder('native')																	
																			
		# call cfs to get a command factory																	
		#cmd = cfs.command(cfs.CommandCodes.TO_OUTPUT_ENABLE_CC)																	
		cmd = cfs.get_command_builder(cfs.CommandCodes.TO_OUTPUT_ENABLE_CC)																	
		# set parameters																	
		cmd.set('dest_IP', 'localhost')																	
																			
		# send the command to enable TLM transmits																	
		cmd.send()																	
		self.wait(10)																	
																			
		# verify the FSW sent TLM in response to the enable TLM command sent																	
		evr_count = cfs.count_sample(cfs.Telemetry.CFE_EVS_LONG_EVENT_TLM)																	
		orig_evr_count = evr_count																	
		self.assertTrue(evr_count > 0)																	
		while (evr_count > 0):																	
		self.wait(0.1)																	
		evr_hk = cfs.decom_sample(cfs.Telemetry.CFE_EVS_LONG_EVENT_TLM, evr_count-1)																	
		self.print("EVR",evr_count,"timestamp in seconds:", evr_hk['timestamp']['seconds'],",subseconds:",evr_hk['timestamp']['sub_sec'], sep=' ')																	
		self.print("EVR",evr_count,"is from App:", evr_hk['Payload']['PacketID']['AppName'], sep=' ')																	
		self.print("EVR",evr_count,"Event ID is:", evr_hk['Payload']['PacketID']['EventID'], sep=' ')																	
		self.print("EVR",evr_count,"Event type is:", evr_hk['Payload']['PacketID']['EventType'], sep=' ')																	
		self.print("EVR",evr_count,"Event is from spacecraft with ID:", evr_hk['Payload']['PacketID']['SpacecraftID'], sep=' ')																	
		self.print("EVR",evr_count,"Event is from processor with ID:", evr_hk['Payload']['PacketID']['ProcessorID'], sep=' ')																	
		self.print("EVR",evr_count,"message is: >>>>>>>>>>>>>>>>>>>>", evr_hk['Payload']['Message'], sep=' ')																	
		evr_count = evr_count - 1																	
		self.print("-----------------------------------------------------------------")																	
		self.print("***** Number of EVRs is:",orig_evr_count, sep=' ')													



