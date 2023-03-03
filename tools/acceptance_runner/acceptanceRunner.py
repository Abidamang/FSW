																
		#!/usr/bin/env python3								
										
		# /******************************************************************************								
		# *                       						
		# ******************************************************************************								
		# * FILE NAME: acceptanceRunner.py								
		# * MODULE: cfs/tools/acceptance_runner/								
		# *								
		# * 								
		# ******************************************************************************								
		# * 							
		# * 								
		# ******************************************************************************								
		# * DESCRIPTION:  								
		# * This file contains the acceptanceRunner tool and support classes.								
		# ******************************************************************************/								
										
		from abc import ABC, abstractmethod								
		import argparse								
		import subprocess								
		from pathlib import Path, PurePath								
		import os								
		import os.path								
		import signal								
		import unittest								
		import json								
		import sys								
		import inspect								
		import importlib.util								
		# Python 3.10 from collections.abc import Sequence								
		from typing import Dict, Tuple, Sequence, List, Type, NewType								
		from collections import namedtuple, deque, defaultdict								
		import time								
		import socket								
		import threading								
		import ctypes								
		import enum								
		import struct								
		import pprint								
		from functools import reduce								
		import locale								
		import warnings								
		import traceback								
										
		################################################################################								
		Nanoseconds = NewType('Nanoseconds', int)								
		Seconds = NewType('Seconds', float)								
										
		class StandardClock:								
		    '''								
		    Use either this StandardClock or a simulated time clock instead of								
		    the actual clock.								
		    '''								
		    def get_time(self) -> Nanoseconds:								
		        #return Nanoseconds(time.time_ns())								
		        ns = time.time() * 1e9								
		        return Nanoseconds(int(ns))								
		    def wait(self, s: Seconds) -> None:								
		        time.sleep(s)								
										
		class MockClock(StandardClock):								
		    _time: Nanoseconds = 0								
		    def set_time(self, t: Nanoseconds) -> None:								
		        self._time = t								
		    def get_time(self) -> Nanoseconds:								
		        return self._time								
		    def wait(self, s: Seconds) -> None:								
		        self._time += Nanoseconds(s*1e9)								
										
		class MockActionClock(MockClock):								
		    def __init__(self, actions=[]) -> None:								
		        self.actions = actions								
		        self.idx = 0								
		    def set_new_actions(self, actions=[]) -> None:								
		        self.actions = actions								
		        self.idx = 0								
		    def wait(self, s: Seconds) -> None:								
		        super().wait(s)								
		        if self.idx < len(self.actions):								
		            if self.actions[self.idx]:								
		                self.actions[self.idx](self._time) # invoke the lambda								
		            self.idx += 1								
										
		class TestMockActionClock(unittest.TestCase):								
		    def test_no_actions(self) -> None:								
		        uut = MockActionClock()								
		        uut.wait(3)								
		        self.assertEqual(3*1e9, uut.get_time())								
		    def inc(self, amount=1) -> None:								
		        self.x += amount								
		    def test_first_action(self) -> None:								
		        self.x = 0								
		        uut = MockActionClock([lambda _: self.inc()])								
		        self.assertEqual(0, self.x)								
		        uut.wait(3)								
		        self.assertEqual(1, self.x)								
		    def test_action_nothing_action(self) -> None:								
		        self.x = 0								
		        uut = MockActionClock([lambda _: self.inc(2), None, lambda _: self.inc(7)])								
		        self.assertEqual(0, self.x)								
		        uut.wait(3)								
		        self.assertEqual(2, self.x)								
		        uut.wait(3)								
		        self.assertEqual(2, self.x)								
		        uut.wait(3)								
		        self.assertEqual(9, self.x)								
		        uut.wait(3)								
		        self.assertEqual(9, self.x)								
										
										
		#TODO(djk): create a clock with a defined start date								
		# and time that advances in real time								
										
										
		################################################################################								
		class OutputLevel(enum.IntEnum):								
		    SILENT = 0								
		    NORMAL = 1								
		    VERBOSE = 2								
		    SPAM = 3								
										
		class OutputMixin:								
		    output_level = OutputLevel.NORMAL								
		    _pp = pprint.PrettyPrinter(indent=4)								
		    def verbose(self) -> None:								
		        self.set_output_level(OutputLevel.VERBOSE)								
		    def silence(self) -> None:								
		        self.set_output_level(OutputLevel.SILENT)								
		    def set_output_level(self, output_level: OutputLevel) -> None:								
		        #print(f'Setting {self} to output level {output_level}')								
		        self.output_level = output_level								
		        members = inspect.getmembers(self, lambda o: isinstance(o, OutputMixin))								
		        for name, value in members:								
		            value.set_output_level(output_level)								
		    def print_error(self, msg: str) -> None:								
		        # future options								
		        print(msg, flush=True)								
		    def print_normal(self, msg: str) -> None:								
		        if self.output_level >= OutputLevel.NORMAL:								
		            print(msg, flush=True)								
		    def print_verbose(self, msg: str) -> None:								
		        if self.output_level >= OutputLevel.VERBOSE:								
		            print(msg, flush=True)								
		    def print_spam(self, msg: str) -> None:								
		        if self.output_level >= OutputLevel.SPAM:								
		            print(msg, flush=True)								
		    def hex_print(self, data: bytearray, preamble: str='') -> None:								
		        if self.output_level >= OutputLevel.VERBOSE:								
		            hex_print(data, preamble)								
		    def pprint(self, data) -> None:								
		        self._pp.pprint(data)								
		    def pformat(self, data) -> str:								
		        return self._pp.pformat(data)								
		    def print_call_location(self, **args) -> None:								
		        yes_print = args.get('force_print', False) or self.output_level >= OutputLevel.NORMAL								
		        stack_offset = args.get('stack_offset', 2) + 1 # skip this level too								
		        prefix = args.get('prefix', '')								
		        line_list = args.get('alt_tb', traceback.format_stack(limit=stack_offset)[0])								
		        if yes_print:								
		            for l in line_list.splitlines():								
		                print(f'{prefix}{l}')								
										
		def hex_print(data: bytearray, preamble: str='') -> None:								
		    print(preamble, end='', flush=True)								
		    for b in data:								
		        print(f"0x{format(b, '02X')}", end=" ")								
		    print(flush=True)								
										
										
		class TestOutputMixin(unittest.TestCase):								
		    class ClassA:								
		        pass								
		    class ClassB(OutputMixin):								
		        pass								
		    class ClassC(OutputMixin):								
		        def __init__(self) -> None:								
		            self.a = TestOutputMixin.ClassA()								
		            self.b = TestOutputMixin.ClassB()								
		    def test_set_output_level(self) -> None:								
		        test_obj = TestOutputMixin.ClassC()								
		        self.assertEqual(test_obj.output_level, OutputLevel.NORMAL)								
		        self.assertEqual(test_obj.b.output_level, OutputLevel.NORMAL)								
		        test_obj.set_output_level(OutputLevel.SPAM)								
		        self.assertEqual(test_obj.output_level, OutputLevel.SPAM)								
		        self.assertEqual(test_obj.b.output_level, OutputLevel.SPAM)								
										
										
		################################################################################								
		# PyPi has a bitarray class that does some of what I need, but not the								
		# most important things. Add a dependency plus code, or just add code?								
										
		class BitMath:								
		    '''								
		    Support routines for BitFields.								
		    '''								
		    def __init__(self, start_bit: int, bit_length: int) -> None:								
		        assert start_bit >= 0								
		        assert bit_length > 0								
		        self.start_bit = start_bit								
		        self.bit_length = bit_length								
		        self.last_bit = start_bit + bit_length								
		        self.byte_first = start_bit//8								
		        self.byte_last = (self.last_bit+7)//8								
		        self.lsb_shift = (self.byte_last*8 - self.last_bit)								
		        self.mask = 2**bit_length - 1								
		        self.bit_window = 8 - (self.byte_last*8 - self.last_bit)								
		        self.bytes_to_process = self.byte_last - self.byte_first								
		    def trim(self, value: int) -> int:								
		        return value & ((1 << self.bit_window) - 1)								
		    def start_of_last_byte(self) -> int:								
		        return (self.byte_last - 1) * 8								
		    def extend_sign(self, value: int) -> int:								
		        if (value < 0):								
		            return value + 2**self.bit_length								
		        else:								
		            return value								
		    def occupies_single_byte(self) -> bool:								
		        return self.byte_first == (self.byte_last - 1)								
										
		    def start_multi_byte_insertion(self, value) -> None:								
		        '''Start at the LSB end and work towards MSB.'''								
		        self.bit_window = 8 - (self.byte_last*8 - self.last_bit)								
		        self.current_insertion_bit = self.start_of_last_byte()								
		        assert 1 <= self.bit_window <= 8								
		        self.insertion_value = value								
		    def bytes_remain(self) -> bool:								
		        return self.bytes_to_process > 0								
		    def insert_location(self) -> int:								
		        return self.current_insertion_bit								
		    def insert_size(self) -> int:								
		        return self.bit_window								
		    def insert_value(self) -> int:								
		        return self.trim(self.insertion_value)								
		    def next_byte(self) -> None:								
		        self.insertion_value >>= self.bit_window								
		        self.bit_length -= self.bit_window								
		        self.bytes_to_process -= 1								
		        if self.bytes_to_process == 1:								
		            # set the window size to the remaining bit size								
		            self.bit_window = self.bit_length								
		        else:								
		            # extract a full byte next pass								
		            self.bit_window = 8								
		        self.current_insertion_bit -= self.bit_window								
										
		class BitCursor:								
		    def __init__(self, start: int=0) -> None:								
		        self.bit_position = start								
		    def offset(self) -> int:								
		        return self.bit_position								
		    def advance(self, bits: int=1) -> None:								
		        self.bit_position += bits								
										
		class BitFields:								
		    '''								
		    Treat bits as an array where bit fields can be converted to								
		    unsigned and signed integers.								
		    '''								
		    def __init__(self, arg) -> None:								
		        if isinstance(arg, str):								
		            required_bytes = (len(arg)+7)//8								
		            pad_bits = (required_bytes*8 - len(arg))								
		            value = int(arg, 2) << pad_bits # pad on lsb								
		            self.data = bytearray(value.to_bytes(length=required_bytes,								
		                                                 byteorder='big'))								
		        elif isinstance(arg, int):								
		            self.data = bytearray(arg)								
		        elif isinstance(arg, bytearray):								
		            self.data = arg # reference to manipulatable buffer								
		    def extract_unsigned(self, start_bit: int, bit_length: int, little_endian: bool) -> int:								
		        bm = BitMath(start_bit, bit_length)								
		        whole_value = int.from_bytes(self.data[bm.byte_first:bm.byte_last],								
		                                 byteorder='big')								
		        whole_value = ((whole_value >> bm.lsb_shift) & bm.mask)								
		        if (bit_length > 8) and (little_endian):								
		            return self._swap(whole_value, bit_length)								
		        return whole_value								
		    def extract_signed(self, start_bit: int, bit_length: int, little_endian: bool) -> int:								
		        '''algorithm from Hacker's Delight (second edition), page 20.'''								
		        unsigned = self.extract_unsigned(start_bit, bit_length, little_endian)								
		        sign_bit_mask = 1 << (bit_length-1)								
		        sans_sign_bit = unsigned & (sign_bit_mask-1)								
		        return sans_sign_bit - (unsigned & sign_bit_mask)								
		    def insert_integer(self,								
		                       start_bit: int,								
		                       bit_length: int,								
		                       value: int,								
		                       little_endian: bool) -> None:								
		        bm = BitMath(start_bit, bit_length)								
		        unsigned_value = bm.extend_sign(value)								
		        if (bit_length > 8) and (little_endian):								
		            unsigned_value = self._swap(unsigned_value, bit_length)								
		        if bm.occupies_single_byte():								
		            self._single_byte_insert(bm, unsigned_value)								
		        else:								
		            self._multi_byte_insert(bm, unsigned_value)								
		    def _swap(self, original: int, bit_length: int) -> int:								
		        the_bytes = original.to_bytes(length=(bit_length+7)//8, byteorder='little')								
		        accumulator = 0								
		        for b in the_bytes:								
		            accumulator = (accumulator << 8) | b								
		        return accumulator								
		    def _single_byte_insert(self, bm: BitMath, value: int) -> None:								
		        right_byte = int(value.to_bytes(length=5,								
		                                        byteorder='big')[-1]) & bm.mask								
		        mask = bm.mask << bm.lsb_shift								
		        insert = (right_byte << bm.lsb_shift) & mask								
		        answer = self.data[bm.byte_first] & (~mask) | insert								
		        self.data[bm.byte_first] = answer								
		    def _multi_byte_insert(self, bm: BitMath, value: int) -> None:								
		        bm.start_multi_byte_insertion(value)								
		        while bm.bytes_remain():								
		            self.insert_integer(bm.insert_location(),								
		                                bm.insert_size(),								
		                                bm.insert_value(),								
		                                False)								
		            bm.next_byte()								
										
										
										
		class TestBitFields(unittest.TestCase):								
		    def test_str(self) -> None:								
		        uut = BitFields('11001100')								
		        self.assertEqual(1, len(uut.data))								
		    def test_int(self) -> None:								
		        uut = BitFields(3)								
		        self.assertEqual(3, len(uut.data))								
		        self.assertEqual(0, uut.extract_unsigned(3, 12, False))								
		    def test_extract_unsigned_within_byte(self) -> None:								
		        uut = BitFields('11001101')								
		        self.assertEqual(0xC, uut.extract_unsigned(0, 4, False))								
		        self.assertEqual(0xD, uut.extract_unsigned(4, 4, False))								
		    def test_extract_unsigned_across_bytes(self) -> None:								
		        uut = BitFields('110010111010')								
		        self.assertEqual(0xC, uut.extract_unsigned(0, 4, False))								
		        self.assertEqual(0xB, uut.extract_unsigned(4, 4, False))								
		        self.assertEqual(0b10111010, uut.extract_unsigned(4, 8, False))								
		    def test_bytearray(self) -> None:								
		        uut = BitFields(bytearray([0, 0xFF, 0]))								
		        self.assertEqual(3, len(uut.data))								
		        self.assertEqual(0b11100000000, uut.extract_unsigned(13, 11, False))								
		    def test_extract_signed_across_bytes(self) -> None:								
		        uut = BitFields('11001011101010000111')								
		        self.assertEqual(-9, uut.extract_signed(4, 5, False))								
		        self.assertEqual(58, uut.extract_signed(5, 7, False))								
		    def test_insert_integer_within_byte(self) -> None:								
		        uut = BitFields('111111110111111011111111')								
		        self.assertEqual(0b01111110, uut.data[1])								
		        uut.insert_integer(9, 6, 4, False)								
		        self.assertEqual(0b00001000, uut.data[1])								
		        uut.insert_integer(9, 6, -1, False)								
		        self.assertEqual(0b01111110, uut.data[1])								
		        uut.insert_integer(8, 1, 1, False)								
		        self.assertEqual(0b11111110, uut.data[1])								
		        uut.insert_integer(15, 1, 1, False)								
		        self.assertEqual(0b11111111, uut.data[1])								
		    def test_insert_integer_across_bytes(self) -> None:								
		        uut = BitFields(9)								
		        uut.insert_integer(11, 18, 0b11_0011_0011_1010_1111, False)								
		        self.assertEqual(0b000_11_001, uut.data[1])								
		        self.assertEqual(0b1_0011_101, uut.data[2])								
		        self.assertEqual(0b0_1111_000, uut.data[3])								
		    def test_negative_insert_integer_across_bytes(self) -> None:								
		        uut = BitFields(9)								
		        uut.insert_integer(26, 24, -1, False)								
		        self.assertEqual(0b0011_1111, uut.data[3])								
		        self.assertEqual(0b1111_1111, uut.data[4])								
		        self.assertEqual(0b1111_1111, uut.data[5])								
		        self.assertEqual(0b1100_0000, uut.data[6])								
		    def test_odd_looking(self) -> None:								
		        uut = BitFields(9)								
		        uut.insert_integer(40, 2, 3, False)								
		        self.assertEqual(0b1100_0000, uut.data[5])								
		        uut.insert_integer(46, 2, 3, False)								
		        self.assertEqual(0b1100_0011, uut.data[5])								
		    def test_insert_little_endian_integer(self) -> None:								
		        buffer = bytearray([0b0000_0001, 0, 0, 0b1000_0000])								
		        uut = BitFields(buffer)								
		        uut.insert_integer(8, 16, 0b1111_0011_1100_1111, little_endian=True)								
		        self.assertEqual(0b0000_0001, uut.data[0])								
		        self.assertEqual(0b1100_1111, uut.data[1])								
		        self.assertEqual(0b1111_0011, uut.data[2])								
		        self.assertEqual(0b1000_0000, uut.data[3])								
		    def test_insert_little_endian_integer_offset(self) -> None:								
		        '''Does this make sense????'''								
		        buffer = bytearray([0b0001_0001, 0, 0, 0b1000_1000])								
		        uut = BitFields(buffer)								
		        #print(f'Inserting 0x{0b1011_0011_0101_1010_1100_1101:08X}')								
		        #hex_print(buffer, 'before: ')								
		        uut.insert_integer(4, 24, 0b1011_0011_0101_1010_1100_1101, little_endian=True)								
		        #hex_print(buffer, 'after: ')								
		        self.assertEqual(0b0001_1100, uut.data[0])								
		        self.assertEqual(0b1101_0101, uut.data[1])								
		        self.assertEqual(0b1010_1011, uut.data[2])								
		        self.assertEqual(0b0011_1000, uut.data[3])								
		    def test_extract_little_endian_unsigned_across_bytes(self) -> None:								
		        uut = BitFields('110010111010111100001010')								
		        self.assertEqual(0b1100, uut.extract_unsigned(0, 4, little_endian=True))								
		        self.assertEqual(0b1011, uut.extract_unsigned(4, 4, little_endian=True))								
		        self.assertEqual(0b10111010, uut.extract_unsigned(4, 8, little_endian=True))								
		        self.assertEqual(0b1010111111001011, uut.extract_unsigned(0, 16, little_endian=True))								
		        self.assertEqual(0b1111000010111010, uut.extract_unsigned(4, 16, little_endian=True))								
										
										
										
		################################################################################								
		class CfsMidVersion(enum.IntEnum):								
		    v1short = 0     # 11-bit CCSDS Primary Header APID								
		    v1 = 1          # 16-bit MID including first 5 bits of the CCSDS Primary Header and APID								
		    v2 = 2          # 16-bit MID with some bits of the CCSDS APID and some extended header								
		    vAugustus = 3   # See https://mxrgcc.sharepoint.us/:f:/r/sites/CTO-MaxarCoreAvionicsIRAD/Shared%20Documents/Working%20Group%20Avionics%20Development/03%20-%20Subsystems/02%20-%20FSW/Packet%20Routing?csf=1&web=1&e=8CVeEg								
		    vGateway = 4    # See GP 10098 (32-bit MID)								
		    vTransition = 5 # Temporary version during transition from v1 to vAugustus								
										
										
		################################################################################								
		class NotImplementedYetException(Exception):								
		    pass								
		StorageKey = NewType('StorageKey', int)								
		TypeCode = NewType('TypeCode', int)								
		ApplicationId = NewType('ApplicationId', float)								
		class MagicMid:								
		    '''A helper class to store and convert cFS MIDs.'''								
		    def __init__(self, **args) -> None:								
		        self.command = False								
		        self.app_id = 1								
		        self.subtopic = 0								
		        self.process = 1								
		        self.unit = 1								
		        self._type_code_override(args)								
		        self.command = args.get('command', self.command)								
		        self.app_id = int(args.get('app_id', self.app_id))								
		        self.subtopic = int(args.get('subtopic', self.subtopic))								
		        self.process = int(args.get('process', self.process))								
		        self.unit = int(args.get('unit', self.unit))								
		    def _type_code_override(self, args):								
		        some_type_of_code = any([(tag in args) for tag in ('datagram', 'v1', 'type_code')])								
		        if some_type_of_code:								
		            v1_mid = self._get_mid(args)								
		            self.command = (((v1_mid >> 12) % 2) == 1)								
		            self.app_id = v1_mid % (1<<6)								
		            self.subtopic = (v1_mid >> 6) % (1<<3)								
		            if 'datagram' in args:								
		                process_one = (v1_mid >> 9) % (1<<2)								
		                if (process_one == 0):								
		                    self.process = 1								
		                else:								
		                    self.process = (v1_mid >> 9) % (1<<2)								
		    def _get_mid(self, args):								
		        if 'datagram' in args:								
		            v1_mid = (int(args['datagram'][0]) << 8) | int(args['datagram'][1])								
		        elif 'v1' in args:								
		            v1_mid = int(args['v1'])								
		        elif 'type_code' in args:								
		            v1_mid = int(args['type_code'])								
		        return v1_mid								
		    def get_application_id(self) -> ApplicationId:								
		        return ApplicationId(self.app_id)								
		    def get_subtopic(self) -> int:								
		        return self.subtopic								
		    def set_process(self, p: int) -> None:								
		        self.process = p								
		    def get_process(self) -> int:								
		        return self.process								
		    def get_type_code(self) -> TypeCode:								
		        PROCESS_ONE = 1 << 9								
		        return (self.command << 12) | PROCESS_ONE | (self.subtopic << 6) | self.app_id								
		    def get_storage_key(self) -> StorageKey:								
		        SEC_HDR = 1 << 11								
		        return (self.command << 12) | SEC_HDR | (self.process << 9) | (self.subtopic << 6) | self.app_id								
		    def get_cfs_mid(self, version: CfsMidVersion) -> int:								
		        SEC_HDR = 1 << 11								
		        if (version == CfsMidVersion.v1) or (version == CfsMidVersion.vTransition):								
		            return (self.command << 12) | SEC_HDR | (self.process << 9) | (self.subtopic << 6) | self.app_id								
		        raise NotImplementedYetException()								
		    def is_cmd(self) -> bool:								
		        return self.command								
		    def is_tlm(self) -> bool:								
		        return not self.command								
										
										
		class TestMagicMid(unittest.TestCase):								
		    def test_v1(self) -> None:								
		        uut = MagicMid(v1=0x1808)								
		        self.assertEqual(ApplicationId(8), uut.get_application_id())								
		        self.assertTrue(uut.is_cmd())								
		        self.assertFalse(uut.is_tlm())								
		        self.assertEqual(0x1208, uut.get_type_code())								
		        self.assertEqual(0x1A08, uut.get_storage_key())								
		    def test_v1_legacy_CFE_ES_HK_TLM(self) -> None:								
		        uut = MagicMid(v1=0)								
		        self.assertFalse(uut.is_cmd())								
		        self.assertTrue(uut.is_tlm())								
		        self.assertEqual(0x0200, uut.get_type_code())								
		        self.assertEqual(0x0A00, uut.get_storage_key())								
		    def test_old_style_with_process_2(self) -> None:								
		        uut = MagicMid(v1=0)								
		        self.assertEqual(0x0200, uut.get_type_code())								
		        self.assertEqual(0x0A00, uut.get_storage_key())								
		        uut.set_process(2)								
		        self.assertEqual(0x0200, uut.get_type_code())								
		        self.assertEqual(0x0C00, uut.get_storage_key())								
		        uut.set_process(1)								
		        self.assertEqual(0x0200, uut.get_type_code())								
		        self.assertEqual(0x0A00, uut.get_storage_key())								
		    def test_v1_command(self) -> None:								
		        uut = MagicMid(v1=8, command=True)								
		        self.assertEqual(ApplicationId(8), uut.get_application_id())								
		        self.assertTrue(uut.is_cmd())								
		        self.assertFalse(uut.is_tlm())								
		        self.assertEqual(0x1208, uut.get_type_code())								
		    def test_v1_telemetry(self) -> None:								
		        uut = MagicMid(v1=8, command=False)								
		        self.assertEqual(ApplicationId(8), uut.get_application_id())								
		        self.assertFalse(uut.is_cmd())								
		        self.assertTrue(uut.is_tlm())								
		        self.assertEqual(0x0208, uut.get_type_code())								
		    def test_vAugustus_command(self) -> None:								
		        uut = MagicMid(app_id=5, process=1, subtopic=0, command=True)								
		        self.assertEqual(ApplicationId(5), uut.get_application_id())								
		        self.assertEqual(0x1205, uut.get_type_code())								
		        uut.set_process(2)								
		        self.assertEqual(ApplicationId(5), uut.get_application_id())								
		        self.assertTrue(uut.is_cmd())								
		        self.assertFalse(uut.is_tlm())								
		        self.assertEqual(0x1C05, uut.get_storage_key())								
		        self.assertEqual(0x1205, uut.get_type_code())								
		    def test_vAugustus_telemetry(self) -> None:								
		        uut = MagicMid(app_id=6, subtopic=1)								
		        self.assertEqual(ApplicationId(6), uut.get_application_id())								
		        self.assertEqual(0x0246, uut.get_type_code())								
		        uut.set_process(2)								
		        self.assertEqual(ApplicationId(6), uut.get_application_id())								
		        self.assertEqual(1, uut.get_subtopic())								
		        self.assertEqual(2, uut.get_process())								
		        self.assertEqual(0x0246, uut.get_type_code())								
		        self.assertFalse(uut.is_cmd())								
		        self.assertTrue(uut.is_tlm())								
		    def test_datagram(self) -> None:								
		        uut = MagicMid(datagram=bytearray([0x0A, 0x46]))								
		        self.assertEqual(0x0246, uut.get_type_code())								
		        self.assertEqual(0x0A46, uut.get_storage_key())								
		    def test_error_in_a_real_test(self) -> None:								
		        tracker_mid = MagicMid(type_code=0x000E) # built on type code								
		        self.assertEqual(0x0A0E, tracker_mid.get_storage_key())								
										
										
		################################################################################								
		#TODO(djk): A named tuple wasn't such a great idea. Make class/subclasses								
		TypeInfo = namedtuple('TypeInfo', \								
		                      ('name', \								
		                       'composite', \								
		                       'metatype', \								
		                       'encoding',\								
		                       'endian', \								
		                       'bit_size', \								
		                       'fields', \								
		                       'extension_type', \								
		                       'dimensions'))								
										
		# cFS has the concept of a MID (basically the 1st 16 bits of a packet)								
		# The initial version of the CtDatabase used the concept of "cc0" which is								
		# the v1 MID shifted up 8 bits and the command code grafted into the lower								
		# byte.								
		def application_command_code_to_mid(self):								
		    return self.value >> 8								
		def application_command_code_to_cc(self):								
		    return self.value & 0xFF								
										
										
		class CtDatabase:								
		    '''								
		    Command and Telemetry Database.								
		    '''								
		    FLOAT_ENCODING = {32: '>f', 64: '>d'}								
		    def __init__(self, source: str="ct.json") -> None:								
		        with open(source, "r") as ct_db:								
		            self._db = json.load(ct_db)								
		        # order is important because telemetry and command codes use applications.								
		        self._generate_application_enums()								
		        self._generate_telemetry_enums()								
		        self._generate_command_code_enums()								
										
		    def _generate_application_enums(self) -> None:								
		        application_ids = dict()								
		        for key, value in self._db['applications'].items():								
		            if (isinstance(value, int)):								
		                application_ids[key] = value								
		            else:								
		                application_ids[key] = value["id"]								
		        self.Applications = enum.IntEnum('Applications',								
		                                         application_ids)								
										
		    def _generate_telemetry_enums(self) -> None:								
		        telemetry_packets = dict()								
		        in_use = set()								
		        for key, value in self._db['telemetry'].items():								
		            if isinstance(value, int):								
		                enum_computer = MagicMid(v1=value, command=False)								
		            elif 'id' in value:								
		                enum_computer = MagicMid(v1=int(value['id']), command=False)								
		            else:								
		                app_name = value['application']								
		                app_id = int(self._db['applications'][app_name]['id'])								
		                subtopic = int(value['subtopic'])								
		                enum_computer = MagicMid(app_id=app_id, subtopic=subtopic)								
		            tc = enum_computer.get_type_code()								
		            if tc in in_use:								
		                print(f'DATABASE ERROR! Two telemetry structures resolve to the same type code (0x{tc:04X})')								
		                for k,v in telemetry_packets.items():								
		                    if v == tc:								
		                        other = k								
		                        break								
		                print(f'{other} {key}')								
		            else:								
		                in_use.add(tc)								
		            telemetry_packets[key] = enum_computer.get_type_code()								
		        self.Telemetry = enum.IntEnum('Telemetry',								
		                                      telemetry_packets)								
										
		    def _generate_command_code_enums(self) -> None:								
		        command_packets = dict()								
		        for key, value in self._db['command_codes'].items():								
		            if isinstance(value, dict):								
		                if 'cc0' in value:								
		                    command_packets[key] = value['cc0']								
		                else:								
		                    app_name = value['app']								
		                    app_id = self._get_app_id(app_name)								
		                    subtopic = int(value['subtopic'])								
		                    mid = MagicMid(app_id=app_id, subtopic=subtopic, command=True).get_type_code()								
		                    new_command_code = int(mid << 8) | int(value['cc'])								
		                    command_packets[key] = new_command_code								
		            else:								
		                command_packets[key] = value								
		        self.CommandCodes = enum.IntEnum('CommandCodes',								
		                                         command_packets)								
		        self.CommandCodes.mid = application_command_code_to_mid								
		        self.CommandCodes.cc = application_command_code_to_cc								
										
		    def _get_app_id(self, app_name: str) -> ApplicationId:								
		        return ApplicationId(int(self._db['applications'][app_name]['id']))								
		    def _get(self, type_name: str) -> TypeInfo:								
		        if type_name in self._db['types']:								
		            t = self._db['types'][type_name]								
		        else:								
		            # Treat command parameters as structures								
		            t = self._db['command_parameters'][type_name]								
		        return TypeInfo(type_name,								
		                        (t['basis'] == 'structure') or (t['basis'] == 'array'),								
		                        t['basis'],								
		                        t.get('encoding', 'composite'),								
		                        t.get('endian', 'big'),								
		                        int(t.get('bit_size', 0)),								
		                        t.get('fields', t.get('values', [])),								
		                        t.get('type', t.get('extends', '')),								
		                        t.get('dimensions', []))								
										
		    def unpack(self, type_name: str, raw: bytearray):								
		        '''								
		        Take raw bytes containing data of type type_name and								
		        return either the unpacked value or a dictionary of								
		        unpacked values (keyed by field name).								
		        '''								
		        typ = self._get(type_name)								
		        cursor = BitCursor()								
		        bits = BitFields(raw)								
		        return self._unpack(typ, cursor, bits)								
										
		    def _unpack(self,								
		                typ: TypeInfo,								
		                cursor: BitCursor,								
		                bits: BitFields):								
		        if typ.composite:								
		            return self._unpack_composite(typ, cursor, bits)								
		        else:								
		            return self._unpack_primitive(typ, cursor, bits)								
		    def _unpack_composite(self,								
		                          typ: TypeInfo,								
		                          cursor: BitCursor,								
		                          bits: BitFields):								
		        assert typ.composite								
		        if typ.metatype == 'structure':								
		            fields = {}								
		            if typ.extension_type != '':								
		                fields = self._unpack(self._get(typ.extension_type), cursor, bits)								
		            for field in typ.fields:								
		                type_name = field['type']								
		                field_typ = self._get(type_name)								
		                fields[field['name']] = self._unpack(field_typ, cursor, bits)								
		            return fields								
		        elif typ.metatype == 'array':								
		            elements = []								
		            type_name = typ.extension_type								
		            element_typ = self._get(type_name)								
		            repeat_count = typ.dimensions[0]								
		            for i in range(repeat_count):								
		                elements.append(self._unpack(element_typ, cursor, bits))								
		            return elements								
										
		    def _unpack_primitive(self,								
		                          typ: TypeInfo,								
		                          cursor: BitCursor,								
		                          bits: BitFields):								
		        assert not typ.composite								
		        if typ.metatype == "integer":								
		            if typ.encoding == "unsigned":								
		                value = bits.extract_unsigned(cursor.offset(), typ.bit_size, little_endian=(typ.endian == 'little'))								
		            else:								
		                value = bits.extract_signed(cursor.offset(), typ.bit_size, little_endian=(typ.endian == 'little'))								
		        elif typ.metatype == 'floating_point':								
		            int_holding_float = bits.extract_unsigned(cursor.offset(),								
		                                                      typ.bit_size, little_endian=(typ.endian == 'little'))								
		            value = self._unpack_float \								
		                (typ.bit_size,								
		                 int_holding_float.to_bytes(length = typ.bit_size//8,								
		                                            byteorder = 'big'))								
		        elif typ.metatype == 'string':								
		            byte_size = typ.bit_size//8								
		            hold = bytearray(byte_size)								
		            for i in range(byte_size):								
		                hold[i] = bits.extract_unsigned(cursor.offset()+(i*8), 8, False)								
		            value = hold.decode(encoding='ascii').rstrip('\x00')								
		        elif typ.metatype == 'enumeration':								
		            int_holding_enum = bits.extract_unsigned(cursor.offset(), typ.bit_size, little_endian=(typ.endian == 'little'))								
		            value = str(int_holding_enum)								
		            for v in typ.fields:								
		                if typ.fields[v] == int_holding_enum:								
		                    value = v								
		                    break								
		        else:								
		            raise Exception(f'Unrecognized primitive metatype "{typ.metatype}".')								
		        cursor.advance(typ.bit_size)								
		        return value								
										
		    def _unpack_float(self, bit_size: int, raw: bytearray) -> float:								
		        fmt = self.FLOAT_ENCODING[bit_size]								
		        siz = (bit_size+7)//8								
		        return struct.unpack(fmt, raw[:siz])[0]								
										
										
		    def pack(self, type_name: str, buffer: bytearray, instructions) -> None:								
		        '''								
		        Using instructions, pack data into buffer as defined by type_name.								
		        '''								
		        typ = self._get(type_name)								
		        cursor = BitCursor()								
		        bits = BitFields(buffer)								
		        self._pack(typ, cursor, bits, instructions)								
										
		    def _pack(self,								
		              typ: TypeInfo,								
		              cursor: BitCursor,								
		              bits: BitFields,								
		              inst) -> None:								
		        if typ.composite:								
		            self._pack_composite(typ, cursor, bits, inst)								
		        else:								
		            self._pack_primitive(typ, cursor, bits, inst)								
										
		    def _pack_primitive(self, typ: TypeInfo, cursor: BitCursor,								
		                        bits: BitFields, inst) -> None:								
		        assert not typ.composite								
		        #print(f'inserting {typ.metatype}={inst} at offset ' \								
		        #      f'{cursor.offset()} ({typ.bit_size})')								
		        if typ.metatype == "integer":								
		            #TODO(djk): make sure inst is actually an integer								
		            bits.insert_integer(cursor.offset(), typ.bit_size, inst, (typ.endian == 'little'))								
		        elif typ.metatype == 'floating_point':								
		            pass								
		        elif typ.metatype == 'string':								
		            byte_size = typ.bit_size//8								
		            translated = inst.encode(encoding='ascii')								
		            for i in range(byte_size):								
		                if i < len(inst):								
		                    hold = translated[i]								
		                else:								
		                    hold = 0								
		                bits.insert_integer(cursor.offset()+(i*8), 8, hold, False)								
		        else:								
		            raise Exception(f'Unrecognized primitive metatype "{typ.metatype}".')								
		        cursor.advance(typ.bit_size)								
										
		    def _pack_composite(self, typ: TypeInfo, cursor: BitCursor,								
		                        bits: BitFields, inst) -> None:								
		        assert typ.composite								
		        #print(f'_pack_composite inst={inst}')								
		        if typ.metatype == 'structure':								
		            if typ.extension_type != '':								
		                self._pack(self._get(typ.extension_type), cursor, bits, inst)								
		            for field in typ.fields:								
		                #print(f'    working on field {field}')								
		                field_name = field['name']								
		                type_name = field['type']								
		                field_typ = self._get(type_name)								
		                value = inst.get(field_name)								
		                self._pack(field_typ, cursor, bits, value)								
		        elif typ.metatype == 'array':								
		            assert typ.dimensions[0] == len(inst)								
		            element_typ = self._get(typ.extension_type)								
		            for v in inst:								
		                self._pack(element_typ, cursor, bits, v)								
		        else:								
		            raise Exception(f'Unrecognized composite metatype "{typ.metatype}".')								
										
		    def byte_size(self, type_name: str) -> int:								
		        return (self.bit_size(type_name)+7)//8								
										
		    def bit_size(self, type_name: str) -> int:								
		        typ = self._get(type_name)								
		        if typ.composite:								
		            return self._composit_bit_size(typ)								
		        else:								
		            return typ.bit_size								
										
		    def _composit_bit_size(self, typ: TypeInfo) -> int:								
		        assert typ.composite								
		        bit_size = 0								
		        if typ.metatype == 'structure':								
		            if typ.extension_type != '':								
		                bit_size += self.bit_size(typ.extension_type)								
		            for field in typ.fields:								
		                bit_size += self.bit_size(field['type'])								
		        elif typ.metatype == 'array':								
		            element_count = reduce((lambda x, y: x * y), typ.dimensions)								
		            element_size = self.bit_size(typ.extension_type)								
		            bit_size = element_count * element_size								
		        return bit_size								
										
		    def is_composite(self, type_name: str) -> bool:								
		        typ = self._get(type_name)								
		        return typ.composite								
										
		    def command_has_parameters(self, cmd) -> bool:								
		        return cmd.name in self._db['command_parameters']								
										
		    def get_command_parameters(self, cmd):								
		        return self._db['command_parameters'][cmd.name]['fields']								
										
										
		class TestCtDatabase(unittest.TestCase):								
		    HEADER = [0x08, 0x00, 0xC0, 0x06, 0x00, 0x9D,								
		              0x00, 0x0F, 0x46, 0x3E, 0xFF, 0xFC]								
		    def setUp(self) -> None:								
		        self.uut = CtDatabase()								
		    def test_byte_size_primitives(self) -> None:								
		        self.assertEqual(1, self.uut.byte_size('uint8'))								
		        self.assertEqual(2, self.uut.byte_size('uint16'))								
		        self.assertEqual(4, self.uut.byte_size('uint32'))								
		        self.assertEqual(8, self.uut.byte_size('uint64'))								
		        self.assertEqual(1, self.uut.byte_size('int8'))								
		        self.assertEqual(2, self.uut.byte_size('int16'))								
		        self.assertEqual(4, self.uut.byte_size('int32'))								
		        self.assertEqual(8, self.uut.byte_size('int64'))								
		        self.assertEqual(4, self.uut.byte_size('float'))								
		        self.assertEqual(8, self.uut.byte_size('double'))								
		    def test_bit_size(self) -> None:								
		        self.assertEqual(64, self.uut.bit_size('double'))								
		        self.assertEqual(16*8, self.uut.bit_size('TelemetryHeader'))								
		    def test_array_bit_size(self):								
		        self.assertEqual(16*3, self.uut.bit_size('ARRAY_TEST'))								
		    def test_extension_bit_size(self):								
		        self.assertEqual(12*8, self.uut.bit_size('EXTENSION_TEST'))								
		    def test_is_composite(self) -> None:								
		        self.assertFalse(self.uut.is_composite('int64'))								
		        self.assertTrue(self.uut.is_composite('TelemetryHeader'))								
		    def test_byte_size_telemetry_header(self) -> None:								
		        self.assertEqual(16, self.uut.byte_size('TelemetryHeader'))								
		    def test_decode_uint8(self) -> None:								
		        answer = self.uut.unpack('uint8', bytearray([13]))								
		        self.assertEqual(answer, 13)								
		    def test_decode_pi_ish_32(self) -> None:								
		        answer = self.uut.unpack('float', bytearray([0x40, 0x49, 0x0f, 0xdb]))								
		        self.assertEqual(answer, 3.1415927410125732)								
		    def test_decode_pi_ish_64(self) -> None:								
		        answer = self.uut.unpack('double',								
		                                 bytearray([0x40,								
		                                            0x09,								
		                                            0x21,								
		                                            0xFB,								
		                                            0x54,								
		                                            0x44,								
		                                            0x2E,								
		                                            0xEA]))								
		        self.assertEqual(answer, 3.14159265359)								
		    def test_unpack_header(self) -> None:								
		        expected = {'header': {'PCID': {'Version': 0, \								
		                                        'IsCommand': 0, \								
		                                        'SecondaryHeader': 1, \								
		                                        'PacketID': 0}, \								
		                               'Sequence': {'SequenceFlags': 3, \								
		                                            'PacketSequenceNumber': 6}, \								
		                               'PldSizeMinusOne': 157}, \								
		                    'timestamp': {'seconds': 1001022, 'sub_sec': 65532}, \								
		                    'spare': 0}								
		        answer = self.uut.unpack('TelemetryHeader',								
		                                 bytearray(self.HEADER))								
		        self.assertDictEqual(answer, expected)								
		    def test_unpack_string16(self) -> None:								
		        buffer = bytearray(16)								
		        buffer[0:8] = b'localhost'								
		        answer = self.uut.unpack('string16', buffer)								
		        self.assertEqual('localhost', answer)								
		    def test_pack_int(self) -> None:								
		        buffer = bytearray(7)								
		        instructions = 0xaabbccdd								
		        self.uut.pack('uint32', buffer, instructions)								
		        self.assertEqual(0xaa, buffer[0])								
		        self.assertEqual(0xbb, buffer[1])								
		        self.assertEqual(0xcc, buffer[2])								
		        self.assertEqual(0xdd, buffer[3])								
		        self.assertEqual(0x00, buffer[4])								
		    def test_pack_little_endian_int(self) -> None:								
		        buffer = bytearray(7)								
		        instructions = 0xddccbbaa								
		        self.uut.pack('le_uint32', buffer, instructions)								
		        self.assertEqual(0xaa, buffer[0])								
		        self.assertEqual(0xbb, buffer[1])								
		        self.assertEqual(0xcc, buffer[2])								
		        self.assertEqual(0xdd, buffer[3])								
		        self.assertEqual(0x00, buffer[4])								
		    def test_pack_little_endian_int_not_full(self) -> None:								
		        buffer = bytearray(7)								
		        instructions = 0x0A06								
		        self.uut.pack('le_uint32', buffer, instructions)								
		        self.assertEqual(0x06, buffer[0])								
		        self.assertEqual(0x0A, buffer[1])								
		        self.assertEqual(0x00, buffer[2])								
		        self.assertEqual(0x00, buffer[3])								
		        self.assertEqual(0x00, buffer[4])								
		    def test_unpack_little_endian_int(self) -> None:								
		        answer = self.uut.unpack('le_uint32', bytearray([0xAA, 0xBB, 0xCC, 0xDD]))								
		        self.assertEqual(answer, 0xddccbbaa)								
		    def test_pack_smaller_string(self) -> None:								
		        buffer = bytearray(30)								
		        for i in range(len(buffer)):								
		            buffer[i] = i								
		        #hex_print(buffer, "before: ")								
		        self.uut.pack('string16', buffer, '123456')								
		        #hex_print(buffer, "after : ")								
		        self.assertEqual(ord('1'), buffer[0])								
		        self.assertEqual(ord('6'), buffer[5])								
		        self.assertEqual(0x00, buffer[6])								
		        self.assertEqual(0x00, buffer[15])								
		        self.assertEqual(16, buffer[16])								
		    def test_pack_larger_string(self) -> None:								
		        buffer = bytearray(30)								
		        for i in range(len(buffer)):								
		            buffer[i] = i								
		        #hex_print(buffer, "before: ")								
		        self.uut.pack('string16', buffer, '1234561234567890abc')								
		        #hex_print(buffer, "after : ")								
		        self.assertEqual(ord('1'), buffer[0])								
		        self.assertEqual(ord('0'), buffer[15])								
		        self.assertEqual(16, buffer[16])								
		    def test_pack_composite(self) -> None:								
		        buffer = bytearray(6)								
		        instructions = {'Version': 0, 'IsCommand': 1,								
		                        'SecondaryHeader': 1, 'PacketID': 37}								
		        self.uut.pack('PacketIdentifier', buffer, instructions)								
		        self.assertEqual(0x18, buffer[0])								
		        self.assertEqual(0x25, buffer[1])								
		        self.assertEqual(0x00, buffer[2])								
		    def test_application_enum(self) -> None:								
		        # This test assumes the test ct.json where TO_LAB is defined at 128								
		        self.assertEqual(self.uut.Applications.TO_LAB, 128)								
		    def test_Telemetry_enum(self) -> None:								
		        self.assertEqual(self.uut.Telemetry.CFE_ES_HK_TLM, 0x0200)								
		        self.assertEqual(self.uut.Telemetry.CFE_EVS_LONG_EVENT_TLM, 0x0208)								
		        self.assertEqual(self.uut.Telemetry.TEST_APP_5_SUB2_TLM, 0x0285) # new style mid								
		    def test_CommandCodes_enum(self) -> None:								
		        '''New-style command code in the ct.json'''								
		        CFE_ES_CLEAR_ER_LOG_CC = 12								
		        cc = (MagicMid(app_id=self.uut.Applications.CFE_ES, subtopic=0, command=True).get_type_code() << 8) + CFE_ES_CLEAR_ER_LOG_CC								
		        self.assertEqual(self.uut.CommandCodes.CFE_ES_CLEAR_ER_LOG_CC, cc)								
		    def test_command_has_parameters(self) -> None:								
		        self.assertTrue(self.uut.command_has_parameters \								
		                        (self.uut.CommandCodes.TO_REMOVE_PKT_CC))								
		        self.assertFalse(self.uut.command_has_parameters \								
		                        (self.uut.CommandCodes.TO_NOP_CC))								
		    def test_get_command_parameters(self) -> None:								
		        # parameters are a list of dictionaries (each with name and type)								
		        params = self.uut.get_command_parameters \								
		            (self.uut.CommandCodes.CFE_ES_DELETE_CDS_CC)								
		        self.assertEqual(len(params), 1)								
		        self.assertEqual('CdsName', params[0]['name'])								
		        self.assertEqual('string40', params[0]['type'])								
		    def test_unpack_array(self) -> None:								
		        '''cFS EDS has fixed-length arrays (we need variable length arrays)'''								
		        expected = [{'Version': 0, 'IsCommand': 0, 'SecondaryHeader': 1, 'PacketID': 1}, \								
		                    {'Version': 0, 'IsCommand': 0, 'SecondaryHeader': 1, 'PacketID': 2}, \								
		                    {'Version': 0, 'IsCommand': 0, 'SecondaryHeader': 1, 'PacketID': 3}]								
		        raw = [0x08, 0x01, 0x08, 0x02, 0x08, 0x03]								
		        answer = self.uut.unpack('ARRAY_TEST', bytearray(raw))								
		        self.assertListEqual(answer, expected)								
		    def test_pack_array(self) -> None:								
		        buffer = bytearray(7)								
		        instructions = [{'Version': 0, 'IsCommand': 0, 'SecondaryHeader': 1, 'PacketID': 1}, \								
		                        {'Version': 0, 'IsCommand': 0, 'SecondaryHeader': 1, 'PacketID': 2}, \								
		                        {'Version': 0, 'IsCommand': 0, 'SecondaryHeader': 1, 'PacketID': 3}]								
		        expected = bytearray([0x08, 0x01, 0x08, 0x02, 0x08, 0x03, 0x00])								
		        self.uut.pack('ARRAY_TEST', buffer, instructions)								
		        self.assertEqual(expected, buffer)								
		    def test_unpack_extension(self) -> None:								
		        '''cFS EDS has extension structures that (optionally/conditionally) extend other structures'''								
		        expected = {'PCID': {'Version': 0, \								
		                             'IsCommand': 0, \								
		                             'SecondaryHeader': 1, \								
		                             'PacketID': 0}, \								
		                    'Sequence': {'SequenceFlags': 3, \								
		                                 'PacketSequenceNumber': 6}, \								
		                    'PldSizeMinusOne': 157, \								
		                    'timestamp': {'seconds': 1001022, 'sub_sec': 65532}}								
		        answer = self.uut.unpack('EXTENSION_TEST',								
		                                 bytearray(self.HEADER))								
		        self.assertDictEqual(answer, expected)								
		    def test_pack_extension(self) -> None:								
		        buffer = bytearray(len(self.HEADER))								
		        instructions = {'PCID': {'Version': 0, \								
		                                 'IsCommand': 0, \								
		                                 'SecondaryHeader': 1, \								
		                                 'PacketID': 0}, \								
		                        'Sequence': {'SequenceFlags': 3, \								
		                                     'PacketSequenceNumber': 6}, \								
		                        'PldSizeMinusOne': 157, \								
		                        'timestamp': {'seconds': 1001022, 'sub_sec': 65532}}								
		        self.uut.pack('EXTENSION_TEST', buffer, instructions)								
		        self.assertEqual(bytearray(self.HEADER), buffer)								
		    def test_unpack_enum(self):								
		        buffer = bytearray(2)								
		        buffer[0] = 1								
		        answer = self.uut.unpack('bool8', buffer)								
		        self.assertEqual('TRUE', answer)								
		        buffer[0] = 0								
		        answer = self.uut.unpack('bool8', buffer)								
		        self.assertEqual('FALSE', answer)								
		    #TODO(djk): test packing an enum    								
										
										
		################################################################################								
		PacketSample = namedtuple('PacketSample',								
		                          'timestamp datagram')								
		SocketAddress = namedtuple('SocketAddress',								
		                           ('host', 'port'))								
										
		class PacketReceiver(OutputMixin):								
		    def __init__(self,								
		                 queue: deque,								
		                 clock: StandardClock,								
		                 addr: SocketAddress=SocketAddress('localhost', 1235)) -> None:								
		        self._queue = queue								
		        self._clock = clock								
		        self._addr = addr								
		        self._error_count = 0								
		        self._running = True								
		        self.output_level = OutputLevel.SILENT								
		    def signal_shutdown(self) -> None:								
		        self._running = False								
		    def run(self) -> None:								
		        self._start()								
		        while self._running:								
		            self._receive()								
		        self._stop()								
		    def _start(self) -> None:								
		        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)								
		        #TODO(djk): bind() could fail if the socket hasn't closed yet.								
		        # Retry after a brief wait???								
		        self._sock.bind(self._addr)								
		        self._sock.settimeout(0.2)								
		    def _stop(self) -> None:								
		        self._running = False								
		        self._close_socket()								
		    def _close_socket(self) -> None:								
		        try:								
		            self._sock.close()								
		        except:								
		            pass								
		    def _receive(self) -> None:								
		        try:								
		            datagram, source = self._sock.recvfrom(4096)								
		            self._handle(datagram)								
		        except socket.timeout:								
		            pass								
		            # socket.timeout is a subclass of OSError								
		            # socket.error is an alias for OSError								
		            # Must try the subclass before the base class								
		        except socket.error as err:								
		            self._error_count += 1								
		            print(f'socket.error "{err}", {self._addr.host}:{self._addr.port}!')								
		            time.sleep(1)								
		            if self._error_count > 5:								
		                print(f'!!!!!!!!! Shuting down receiver.')								
		                self._running = False								
		    def _handle(self, datagram: bytes) -> None:								
		        sample = PacketSample(self._clock.get_time(), datagram)								
		        self._queue.append(sample)								
										
		class TestPacketReceiver(unittest.TestCase):								
		    def setUp(self) -> None:								
		        self.c = MockClock()								
		        self.q = deque()								
		        self.uut = PacketReceiver(self.q, self.c)								
		    def test_defaults(self) -> None:								
		        self.assertEqual(self.uut._addr, SocketAddress('localhost', 1235))								
		    def test_one_item(self) -> None:								
		        self.c.set_time(100)								
		        self.uut._handle(b'fake packet')								
		        answer = self.q.popleft()								
		        self.assertEqual(answer.timestamp, 100)								
		        self.assertEqual(answer.datagram, b'fake packet')								
		    def test_three_items(self) -> None:								
		        self.c.set_time(100)								
		        self.uut._handle(b'fake packet 1')								
		        self.c.set_time(200)								
		        self.uut._handle(b'fake packet 2')								
		        self.c.set_time(300)								
		        self.uut._handle(b'fake packet 3')								
		        answer = self.q.popleft()								
		        self.assertEqual(answer.timestamp, 100)								
		        self.assertEqual(answer.datagram, b'fake packet 1')								
		        answer = self.q.popleft()								
		        self.assertEqual(answer.timestamp, 200)								
		        self.assertEqual(answer.datagram, b'fake packet 2')								
		        answer = self.q.popleft()								
		        self.assertEqual(answer.timestamp, 300)								
		        self.assertEqual(answer.datagram, b'fake packet 3')								
										
		################################################################################								
		class PacketSender(OutputMixin):								
		    def __init__(self,								
		                 addr: SocketAddress=SocketAddress('localhost', 1234)) -> None:								
		        self._addr = addr								
		        self._error_count = 0								
		    def start(self) -> None:								
		        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)								
		    def send_bytes(self, datagram: bytes) -> None:								
		        if self.output_level >= OutputLevel.VERBOSE:								
		            hex_print(datagram,								
		                      f'Sending bytes to {self._addr.host}:{self._addr.port}: ')								
		        bytes_sent = self._sock.sendto(datagram, self._addr)								
		        if bytes_sent != len(datagram):								
		            print('ERROR!')								
		            self._error_count += 1								
		    def stop(self) -> None:								
		        self._sock.close()								
										
		class TestPacketSender(unittest.TestCase):								
		    def try_udp(self) -> PacketSample:								
		        c = StandardClock()								
		        q = deque()								
		        ADDR = SocketAddress('localhost', 6565)								
		        rec = PacketReceiver(q, c, ADDR)								
		        t = threading.Thread(target=rec.run)								
		        t.start()								
		        # Do not sleep here, it causes a later test to fail if this one fails								
		        uut = PacketSender(ADDR)								
		        uut.start()								
		        uut.send_bytes(b'This is the test message. Thank you for playing.')								
		        rec.signal_shutdown()								
		        uut.stop()								
		        t.join()								
		        #TODO(djk): UDP sometimes fails and q is empty and the test fails								
		        answer = q.popleft() # sometimes fails								
		        return answer								
		    def test_udp(self) -> None:								
		        try:								
		            answer = self.try_udp() # sometimes fails								
		            self.assertEqual(answer.datagram,								
		                             b'This is the test message. Thank you for playing.')								
		        except:								
		            pass								
										
		################################################################################								
		class CcsdsPacketHeader(ctypes.BigEndianStructure):								
		    _fields_ = [('packet_version', ctypes.c_uint, 3),								
		                ('is_command', ctypes.c_uint, 1),								
		                ('has_secondary_header', ctypes.c_uint, 1),								
		                ('apid', ctypes.c_uint, 11),								
		                ('sequence_flags', ctypes.c_uint, 2),								
		                ('sequence_count', ctypes.c_uint, 14),								
		                ('length', ctypes.c_ushort, 16)]								
										
		class CcsdsSequenceFlags(enum.IntEnum):								
		    CONTINUATION = 0								
		    FIRST_SEGMENT = 1								
		    LAST_SEGMENT = 2								
		    UNSEGMENTED = 3								
										
		def compute_cfs_checksum(data: bytearray) -> int:								
		    '''logic xor of 0xFF and all the bytes.'''								
		    checksum = 0xFF								
		    for b in data:								
		        #print(f"0x{format(b, '02X')}", end="^")								
		        checksum ^= b								
		    #print(f"0xFF=0x{format(checksum, '02X')}")								
		    return checksum&0xFF								
										
										
		################################################################################								
		class CommandPacket(ABC):								
		    @abstractmethod								
		    def to_bytes(self) -> bytearray:								
		        return bytearray(0)								
										
		class ParameterlessCommandPacket(CommandPacket):								
		    '''***This assumes cFS v2 header!***'''								
		    def __init__(self, mid: int, command_code: int, sequence: int) -> None:								
		        self._mid = mid								
		        self._command_code = command_code								
		        self._sequence = sequence								
		        C_TRUE = 1								
		        self.header = CcsdsPacketHeader(packet_version=0, is_command=C_TRUE,								
		            has_secondary_header=C_TRUE, apid=mid,								
		            sequence_flags=int(CcsdsSequenceFlags.UNSEGMENTED),								
		            sequence_count=sequence, length=9)								
		    def to_bytes(self) -> bytearray:								
		        answer = bytearray(16)								
		        answer[:6] = bytearray(self.header)[:6]								
		        answer[6] = 8								
		        answer[7] = 0								
		        answer[8] = 0								
		        answer[9] = 0								
		        answer[10] = self._command_code								
		        answer[11] = compute_cfs_checksum(answer[:7])								
		        answer[12:16] = [0, 0, 0, 0]								
		        return answer								
										
										
		class TestParameterlessCommandPacket(unittest.TestCase):								
		    def test_all_zeros(self) -> None:								
		        uut = ParameterlessCommandPacket(0, 0, 0)								
		        b = uut.to_bytes()								
		        #hex_print(b)								
		        self.assertEqual(8, len(b))								
		        VERSION = '000'								
		        CMD = '1'								
		        SEC_HDR = '1'								
		        APP_ID_UPPER = '000' # upper 3 bits of 11								
		        self.assertEqual \								
		            (b[0], int(''.join([VERSION, CMD, SEC_HDR, APP_ID_UPPER]), 2))								
		        self.assertEqual(b[1], 0) # lower 8 bits of app id								
		        self.assertEqual(b[2], int(CcsdsSequenceFlags.UNSEGMENTED) << (14-8))								
		        self.assertEqual(b[3], 0)								
		        self.assertEqual(b[4], 0)								
		        SEC_HDR_LENGTH = 2								
		        PAYLOAD_LENGTH = SEC_HDR_LENGTH - 1								
		        self.assertEqual(b[5], PAYLOAD_LENGTH)								
		        self.assertEqual(b[6], 0) # command code								
		        self.assertEqual(b[7], compute_cfs_checksum(b[:7]))								
		    def test_non_zero_command(self) -> None:								
		        uut = ParameterlessCommandPacket(307, 3, 0x2ABC)								
		        b = uut.to_bytes()								
		        #hex_print(b)								
		        self.assertEqual(8, len(b))								
		        VERSION = '000'								
		        CMD = '1'								
		        SEC_HDR = '1'								
		        APP_ID_UPPER = '001' # upper 3 bits of 11								
		        self.assertEqual \								
		            (b[0], int(''.join([VERSION, CMD, SEC_HDR, APP_ID_UPPER]), 2))								
		        self.assertEqual(b[1], 0x33) # lower 8 bits of app id								
		        self.assertEqual \								
		            (b[2],								
		             int(CcsdsSequenceFlags.UNSEGMENTED) << (14-8) | (0x2ABC >> 8))								
		        self.assertEqual(b[3], 0x2ABC & 0xFF)								
		        self.assertEqual(b[4], 0)								
		        SEC_HDR_LENGTH = 2								
		        PAYLOAD_LENGTH = SEC_HDR_LENGTH - 1								
		        self.assertEqual(b[5], PAYLOAD_LENGTH)								
		        self.assertEqual(b[6], 3) # command code								
		        self.assertEqual(b[7], compute_cfs_checksum(b[:7]))								
										
										
										
		################################################################################								
		class CommandBuilder(CommandPacket):								
		    '''								
		    Base class for ExternalEntity-specific commands.								
		    '''								
		    def __init__(self, db: CtDatabase, cmd) -> None:								
		        self._mid = cmd.mid()								
		        self._cc = cmd.cc()								
		        self.name = cmd.name								
		        self._db = db								
		        if db.command_has_parameters(cmd):								
		            self._params = db.get_command_parameters(cmd)								
		        else:								
		            self._params = None								
		        self._process = 1								
		    def is_valid(self) -> bool:								
		        if self._params is None:								
		            return True								
		        for i in range(len(self._params)):								
		            if 'value' not in self._params[i]:								
		                return False								
		        return True								
										
		    def to_bytes(self) -> bytearray:								
		        if self._params is None:								
		            return ParameterlessCommandPacket(self._mid, self._cc, 0).to_bytes()								
		        byte_count = self._db.byte_size(self.name)								
		        payload = bytearray(byte_count)								
		        instructions = {}								
		        for f in self._params:								
		            instructions[f['name']] = f['value']								
		        self._db.pack(self.name, payload, instructions)								
		        C_TRUE = 1								
		        sec_hdr_siz = 10 # cFS v2-specific size								
		        header = CcsdsPacketHeader(packet_version=0, is_command=C_TRUE,								
		            has_secondary_header=C_TRUE, apid=self._mid,								
		            sequence_flags=int(CcsdsSequenceFlags.UNSEGMENTED),								
		            sequence_count=0, length=byte_count+sec_hdr_siz-1)								
		        answer = bytearray(16+byte_count)								
		        answer[:6] = bytearray(header)[:6]								
		        answer[6] = 8								
		        answer[7] = 0								
		        answer[8] = 0								
		        answer[9] = 0								
		        answer[10] = self._cc								
		        for i in range(byte_count):								
		            answer[i+16] = payload[i]								
		        answer[12:16] = [0, 0, 0, 0]								
		        answer[11] = compute_cfs_checksum(answer)								
		        return answer								
										
		    def set(self, key, value):								
		        for i in range(len(self._params)):								
		            if self._params[i]['name'] == key:								
		                #TODO: raise an error if value is the wrong type								
		                self._params[i]['value'] = value								
		                break								
		        #TODO: raise an error if key doesn't name a parameter								
		        return self								
										
		    def set_transponder(self, t):								
		        self._transponder = t								
		        return self								
		    def send(self) -> None:								
		        self._transponder.send_bytes(self.to_bytes())								
		        return self								
		    def unit(self) -> int:								
		        return 1 #TODO: need to be able to set the unit too								
		    def process(self) -> int:								
		        return self._process								
		    def set_process(self, p: int):								
		        self._process = p								
		        if (self._mid >> 9) != 0:								
		            # new style mid (transition to vAugustus)								
		            self._mid = (p << 9) + (self._mid & 0x1FF)								
		            #TODO: use MagicMid								
		        return self								
										
		class TestCommandBuilder(unittest.TestCase):								
		    def setUp(self) -> None:								
		        self.db = CtDatabase()								
		    def test_parameterless_command_bytes(self) -> None:								
		        cmd = self.db.CommandCodes.TO_NOP_CC								
		        uut = CommandBuilder(self.db, cmd)								
		        expected = ParameterlessCommandPacket(cmd.mid(), cmd.cc(), 0)								
		        self.assertEqual(expected.to_bytes(), uut.to_bytes())								
		    def test_parameterless_command_valid(self) -> None:								
		        cmd = self.db.CommandCodes.TO_NOP_CC								
		        uut = CommandBuilder(self.db, cmd)								
		        self.assertTrue(uut.is_valid())								
		    def test_invalid_command(self) -> None:								
		        cmd = self.db.CommandCodes.TO_OUTPUT_ENABLE_CC								
		        uut = CommandBuilder(self.db, cmd)								
		        self.assertFalse(uut.is_valid())								
		    def test_set_string(self) -> None:								
		        cmd = self.db.CommandCodes.TO_OUTPUT_ENABLE_CC								
		        uut = CommandBuilder(self.db, cmd)								
		        uut.set('dest_IP', 'localhost')								
		        self.assertTrue(uut.is_valid())								
		    def test_to_bytes(self) -> None:								
		        # This is the cFS v1 header structure for commands								
		        expected = [0x18, 0x80, 0xC0, 0x00, 0x00, 0x11, 0x06, 0xDD, \								
		                    0x6C, 0x6F, 0x63, 0x61, 0x6C, 0x68, 0x6F, 0x73, \								
		                    0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]								
		        cmd = self.db.CommandCodes.TO_OUTPUT_ENABLE_CC								
		        uut = CommandBuilder(self.db, cmd)								
		        uut.set('dest_IP', 'localhost')								
		        self.assertEqual(bytearray(expected), uut.to_bytes())								
		    def test_default_unit(self) -> None:								
		        cmd = self.db.CommandCodes.TO_OUTPUT_ENABLE_CC								
		        uut = CommandBuilder(self.db, cmd)								
		        self.assertEqual(1, uut.unit())								
		    def test_process(self) -> None:								
		        cmd = self.db.CommandCodes.TO_OUTPUT_ENABLE_CC								
		        uut = CommandBuilder(self.db, cmd)								
		        self.assertEqual(1, uut.process())								
		        uut.set_process(2)								
		        self.assertEqual(2, uut.process())								
		    def test_build_with_process_1(self) -> None:								
		        cmd = self.db.CommandCodes.TO_REMOVE_PKT_CC # new style command in the test CTDB								
		        uut = CommandBuilder(self.db, cmd)								
		        uut.set('Stream', 0xA55A)								
		        expected = [0x1A, 0x09, 0xC0, 0x00, 0x00, 0x03, 0x04, 0xD4, 0xA5, 0x5A]								
		        #hex_print(bytearray(expected), '\nexpected: ')								
		        #hex_print(uut.to_bytes(), '  actual: ')								
		        self.assertEqual(bytearray(expected), uut.to_bytes())								
		    def test_build_with_process_2(self) -> None:								
		        cmd = self.db.CommandCodes.TO_REMOVE_PKT_CC # new style command in the test CTDB								
		        uut = CommandBuilder(self.db, cmd)								
		        uut.set_process(2)								
		        uut.set('Stream', 0xA55A)								
		        expected = [0x1C, 0x09, 0xC0, 0x00, 0x00, 0x03, 0x04, 0xD2, 0xA5, 0x5A]								
		        #hex_print(bytearray(expected), '\nexpected: ')								
		        #hex_print(uut.to_bytes(), '  actual: ')								
		        self.assertEqual(bytearray(expected), uut.to_bytes())								
										
										
										
		################################################################################								
		SampleId = NewType('SampleId', int)								
		ROOTDIR = Path(__file__).parent.absolute()								
										
		class PacketArchive(OutputMixin):								
		    MID_MASK = 0xFFFF								
		    def __init__(self, name :str, ctdb: CtDatabase, queue: deque) -> None:								
		        self._name = name								
		        self._queue = queue								
		        self._store = defaultdict(list)								
		        self._ctdb = ctdb								
		        self._total_count = 0								
		    def has_sample(self, id: MagicMid) -> bool:								
		        self._process_from_queue_to_store()								
		        return self.count_sample(id) > 0								
		    def count_sample(self, id: MagicMid) -> int:								
		        self._process_from_queue_to_store()								
		        storage_key = id.get_storage_key()								
		        return len(self._store[storage_key])								
		    def get_last_sample(self, id: MagicMid) -> PacketSample:								
		        self._process_from_queue_to_store()								
		        storage_key = id.get_storage_key()								
		        return self._store[storage_key][-1]								
		    def get_sample(self, id: MagicMid, idx: int = -1) -> PacketSample:								
		        self._process_from_queue_to_store()								
		        storage_key = id.get_storage_key()								
		        return self._store[storage_key][idx]								
		    def decom_sample(self, id: MagicMid, idx: int = -1):								
		        self._process_from_queue_to_store()								
		        storage_key = id.get_storage_key()								
		        return self._decom(bytearray(self._store[storage_key][idx].datagram), id.get_type_code())								
		    def _decom(self, datagram, type_code: TypeCode):								
		        ccsds_header = self._ctdb.unpack('PacketPrimaryHeader', datagram)								
		        #backwards compatibility								
		        field_name = 'PacketID'								
		        if not field_name in ccsds_header['PCID']:								
		            field_name = 'Application'								
		        pkid = ccsds_header['PCID'][field_name]								
		        if ccsds_header['PCID']['IsCommand']:								
		            print(f'expecting telemetry, got command {pkid}')								
		        else:								
		            ccsds_header = self._ctdb.unpack('TelemetryHeader', datagram)								
		            try:								
		                tlm_type = MagicMid(datagram=datagram).get_type_code()								
		                if tlm_type != type_code:								
		                    print(f'Anomaly: datagram would appear to be type code 0x{tlm_type:04X} ({self._to_name(tlm_type)}), higher level requested 0x{type_code:04X} ({self._to_name(type_code)}).')								
		                tlm_id = self._ctdb.Telemetry(tlm_type)								
		                return self._ctdb.unpack(tlm_id.name, datagram)								
		            except:								
		                pass								
		        return ccsds_header								
		    def _to_name(self, tc: TypeCode) -> str:								
		        try:								
		            name = self._ctdb.Telemetry(tc).name								
		        except:								
		            name = "UNKNOWN_TLM"								
		        return name								
		    def process(self) -> None:								
		        self._process_from_queue_to_store()								
		    def total_count(self):								
		        self._process_from_queue_to_store()								
		        return self._total_count								
		    def _process_from_queue_to_store(self) -> None:								
		        while len(self._queue):								
		            sample = self._queue.popleft()								
		            self._add_sample(sample)								
		    def _add_sample(self, sample: PacketSample) -> None:								
		        mid = MagicMid(datagram=bytearray(sample.datagram))								
		        storage_key = mid.get_storage_key()								
		        self._store[storage_key].append(sample)								
		        self._total_count += 1								
		        if self.output_level >= OutputLevel.VERBOSE:								
		            self._immediate_report(sample, mid.get_type_code())								
		    def _immediate_report(self, sample, type_code) -> None:								
		        if self.output_level >= OutputLevel.VERBOSE:								
		            decom = self._decom(bytearray(sample.datagram), type_code)								
		            packet_ln = int(decom['header']['PldSizeMinusOne']) + 7 # hdr + pld								
		            try:								
		                tlm_name = self._ctdb.Telemetry(type_code).name								
		            except:								
		                tlm_name = f'UNRECOGNIZED_0x{type_code:04X}'								
		            print(f'@@@@ {self._name} {sample.timestamp} Received packet 0x{sample.datagram[0]:02X}{sample.datagram[1]:02X}, ' + \								
		                  f'type {tlm_name}, ' \								
		                  f'{len(sample.datagram)} bytes, expected {packet_ln}', flush=True)								
		            if self.output_level > OutputLevel.VERBOSE:								
		                self.pprint(decom)								
										
										
		class TestPacketArchive(unittest.TestCase):								
		    V1_CMD_MID = SampleId(0x1A06)								
		    V1_TLM_MID = SampleId(0x0A00)								
		    def setUp(self) -> None:								
		        self.q = deque()								
		        self._ctdb = CtDatabase(f'{ROOTDIR}/ct.json')								
		        self.uut = PacketArchive('test', self._ctdb, self.q)								
		    def test_empty(self) -> None:								
		        self.assertFalse(self.uut.has_sample(MagicMid(type_code=self.V1_TLM_MID)))								
		    def test_one_packet(self) -> None:								
		        packet = ParameterlessCommandPacket(self.V1_CMD_MID, 0, 0)								
		        sample = PacketSample(37, packet.to_bytes())								
		        self.q.append(sample)								
		        mid = MagicMid(type_code=self.V1_CMD_MID, command=True)								
		        tmp = MagicMid(datagram=packet.to_bytes())								
		        self.assertEqual(mid.get_storage_key(), tmp.get_storage_key())								
		        self.assertEqual(mid.get_type_code(), tmp.get_type_code())								
		        self.assertTrue(self.uut.has_sample(mid))								
		        self.assertEqual(1, self.uut.count_sample(mid))								
		    def test_two_packets(self) -> None:								
		        p1 = ParameterlessCommandPacket(self.V1_CMD_MID, 0, 0)								
		        p2 = ParameterlessCommandPacket(self.V1_CMD_MID, 1, 2)								
		        self.q.append(PacketSample(37, p1.to_bytes()))								
		        self.q.append(PacketSample(38, p2.to_bytes()))								
		        self.assertTrue(self.uut.has_sample(MagicMid(type_code=self.V1_CMD_MID, command=True)))								
		        self.assertEqual(2, self.uut.count_sample(MagicMid(type_code=self.V1_CMD_MID, command=True)))								
		        sample = self.uut.get_last_sample(MagicMid(type_code=self.V1_CMD_MID, command=True))								
		        self.assertEqual(sample.timestamp, 38)								
		        self.assertEqual(sample.datagram, p2.to_bytes())								
		    def test_packet_name_lookup(self):								
		        TO_LAB_HK_TLM = 0x0080 # old-style type code								
		        raw = [0x08, 0x80, 0xC0, 0x02, 0x00, 0x0D, 0x00, 0x0F, 0x46, 0x2F,								
		               0x35, 0x31, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]								
		        sample = PacketSample(37, bytearray(raw))								
		        self.q.append(sample)								
		        self.uut.process()								
		        mid = MagicMid(type_code=TO_LAB_HK_TLM)								
		        self.assertEqual(0x0A80, mid.get_storage_key())								
		        self.assertTrue(self.uut.has_sample(mid))								
		        answer = self.uut.decom_sample(mid)								
		        self.assertEqual(1001007, answer['timestamp']['seconds'])								
		        self.assertEqual(1, answer['Payload']['CommandCounter'])								
										
										
										
										
		################################################################################								
		class TimeoutException(Exception):								
		    pass								
		class PacketTracker(OutputMixin):								
		    def __init__(self, clock: StandardClock, arc: PacketArchive, id: SampleId) -> None:								
		        self.clock = clock								
		        self.arc = arc								
		        self.mid = MagicMid(type_code=id)								
		        self.timeout_seconds = 10								
		    def set_timeout(self, seconds):								
		        self.timeout_seconds = seconds								
		        return self								
		    def set_process(self, process: int):								
		        self.mid.set_process(process)								
		        return self								
										
		    def has_sample(self) -> bool:								
		        return self.arc.has_sample(self.mid)								
		    def samples(self) -> int:								
		        return self.arc.count_sample(self.mid)								
		    def get(self, idx: int = -1) -> PacketSample:								
		        return self.arc.get_sample(self.mid, idx)								
		    def decom(self, idx: int = -1) -> dict:								
		        return self.arc.decom_sample(self.mid, idx)								
		    def wait_for_update(self, count:int = 1):								
		        expected = self.samples() + count								
		        timeout = self.clock.get_time() + (self.timeout_seconds * 1e9)								
		        while self.clock.get_time() < timeout:								
		            if self.samples() >= expected:								
		                return self								
		            self.clock.wait(0.1)								
		        raise TimeoutException(f'Update count {expected} not reached before timeout')								
		    def wait_for_condition(self, condition, sample_credit=1):								
		        sample_to_check = self.samples()								
		        if sample_to_check != 0:								
		            sample_to_check = max(0, sample_to_check - sample_credit)								
		        timeout = self.clock.get_time() + (self.timeout_seconds * 1e9)								
		        while self.clock.get_time() < timeout:								
		            while self.samples() > sample_to_check:								
		                if condition(self.decom(sample_to_check)):								
		                    return self # condition met								
		                sample_to_check += 1								
		            self.clock.wait(0.1)								
		        raise TimeoutException(f'Condition not reached before timeout')								
										
										
		class TestPacketTracker(unittest.TestCase):								
		    ARBITRARY_TLM_MID = SampleId(MagicMid(app_id=5, process=1, subtopic=1).get_type_code()) # 0x0A45								
		    RAW = [0x0A, 0x45, 0xC0, 0x02, 0x00, 0x0D, 0x00, 0x0F, 0x46, 0x2F,								
		           0x35, 0x31, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]								
		    def setUp(self) -> None:								
		        self.q = deque()								
		        self._ctdb = CtDatabase(f'{ROOTDIR}/ct.json')								
		        self.arc = PacketArchive('test', self._ctdb, self.q)								
		        self.clock = MockActionClock([None, None, None])								
		        self.uut = PacketTracker(self.clock, self.arc, self.ARBITRARY_TLM_MID)								
		        self.field_b = 0								
		    def test_empty(self) -> None:								
		        self.assertFalse(self.uut.has_sample())								
		        self.assertEqual(0, self.uut.samples())								
		    def add_sample(self, raw) -> None:								
		        #print(f'Adding 0x{raw[0]:02X}{raw[1]:02X}...')								
		        sample = PacketSample(999, bytearray(raw))								
		        self.q.append(sample)								
		        self.arc.process()								
		    def add_process_1_sample(self) -> None:								
		        raw = [b for b in self.RAW]								
		        raw[-1] = self.field_b								
		        self.add_sample(raw)								
		    def add_process_2_sample(self) -> None:								
		        raw = [b for b in self.RAW]								
		        raw[0] = 0x0C # process 2								
		        self.add_sample(raw)								
		    def test_one_packet(self) -> None:								
		        self.add_process_1_sample()								
		        self.assertTrue(self.uut.has_sample())								
		        self.assertEqual(1, self.uut.samples())								
		    def test_switch_processes(self) -> None:								
		        self.add_process_2_sample()								
		        self.uut.set_process(2)								
		        #print(f'process = {self.uut.mid.process}')								
		        #print(f'storage key = 0x{self.uut.mid.get_storage_key():04X}')								
		        self.assertTrue(self.uut.has_sample())								
		        self.assertEqual(1, self.uut.samples())								
		        self.uut.set_process(1)								
		        #print(f'process = {self.uut.mid.process}')								
		        #print(f'storage key = 0x{self.uut.mid.get_storage_key():04X}')								
		        self.assertFalse(self.uut.has_sample())								
		        self.assertEqual(0, self.uut.samples())								
		    def test_wait_update_timeout(self) -> None:								
		        with self.assertRaises(TimeoutException):								
		            self.uut.wait_for_update()								
		    def test_wait_update_3(self) -> None:								
		        self.uut.set_process(1)								
		        def send(_):								
		            self.add_process_1_sample()								
		        self.clock.set_new_actions([None, send, None, send, None, send, None, send])								
		        self.clock.set_time(0)								
		        self.uut.wait_for_update(3)								
		        self.assertEqual(3, self.uut.samples())								
		        self.assertAlmostEqual(int(6*0.1*1e9), self.clock.get_time())								
		    def test_new_ctdb_schema(self) -> None:								
		        uut1 = PacketTracker(self.clock, self.arc, self._ctdb.Telemetry.TEST_APP_5_SUB1_TLM)								
		        uut2 = PacketTracker(self.clock, self.arc, self._ctdb.Telemetry.TEST_APP_5_SUB2_TLM)								
		        self.assertEqual(0, uut1.samples())								
		        self.assertEqual(0, uut2.samples())								
		    def test_lambda_wait(self) -> None:								
		        uut = PacketTracker(self.clock, self.arc, self._ctdb.Telemetry.TEST_APP_5_SUB1_TLM)								
		        def send(_):								
		            self.field_b += 2								
		            self.add_process_1_sample()								
		        self.clock.set_new_actions([None, send, None, send, None, send, None, send])								
		        def checker(decom: dict) -> bool:								
		            return decom['Payload']['Field_B'] > 3								
		        uut.wait_for_condition(checker)								
		        self.assertEqual(2, self.uut.samples())								
		    def test_stress_condition_wait(self) -> None:								
		        uut = PacketTracker(self.clock, self.arc, self._ctdb.Telemetry.TEST_APP_5_SUB1_TLM)								
		        def send(_):								
		            self.field_b += 1								
		            self.add_process_1_sample()								
		            self.field_b += 1								
		            self.add_process_1_sample()								
		        self.clock.set_new_actions([None, send, None, send, None, send, None, send])								
		        def checker(decom: dict) -> bool:								
		            return decom['Payload']['Field_B'] == 3								
		        uut.wait_for_condition(checker) # will raise TimeoutError if it fails								
										
										
										
										
		################################################################################								
		class RunConfiguration:								
		    '''								
		    Each acceptance test has an "acceptance.json" configuration file.								
		    This class ingests the configuration (possibly from an alternate								
		    source -- for testing) and then provides access methods used deliver								
		    configuration data to the acceptance test classes that need it.								
										
		    The "major magic" of this class is that it converts relative paths								
		    in the configuration JSON into absolute paths relative to the								
		    location of the original JSON file. Thus the configuration file is								
		    the anchor point of the tests it is configuring.								
		    '''								
		    CONFIG_FILE = "acceptance.json"								
		    def __init__(self, where=Path.cwd(), alt_json=None) -> None:								
		        self._base_path = where								
		        self._alt_json = alt_json								
		        self._config_file = os.path.abspath(PurePath(where, self.CONFIG_FILE))								
		        self._configuration = None								
		        self._all_test_files = None								
		        self._load()								
		    def _load(self) -> None:								
		        if self._alt_json is None:								
		            self._load_from_file()								
		        else:								
		            self._load_from_string(self._alt_json)								
		    def _load_from_file(self) -> None:								
		        with open(self._config_file, "r") as config_file:								
		            self._configuration = json.load(config_file)								
		    def _load_from_string(self, string) -> None:								
		        self._configuration = json.loads(string)								
										
		    def files(self) -> Sequence[str]:								
		        self._fix_includes()								
		        return self._all_test_files								
		    def _fix_includes(self) -> None:								
		        self._all_test_files = \								
		            [self._combine(f) for f in self._includes() if self._file_exists(f)]								
		    def _combine(self, f: str) -> str:								
		        return os.path.abspath(PurePath(self._base_path, f))								
		    def _file_exists(self, f: str) -> bool:								
		        return os.path.exists(self._combine(f))								
		    def _includes(self) -> Sequence[str]:								
		        return self._configuration['include']								
										
		    def _search(self) -> None:								
		        '''								
		        Test discovery (so you can add new tests without updating the JSON)								
		        Search all search paths for python files containing AcceptanceSuite								
		        child classes, add them to the self._all_test_files list.								
		        '''								
		        pass #TODO(djk): build&test searching for tests								
										
		    def get_clock_class(self):								
		        if 'clock' in self._configuration:								
		            class_name = self._configuration['clock']								
		            # return the class named								
		            return globals()[class_name]								
		        return StandardClock								
		    def get_ext_ent_value(self, ext_ent: str, key: str):								
		        return self._configuration[ext_ent][key]								
		    def get_ext_ent_path(self, ext_ent: str, key: str):								
		        return self._combine(self._configuration[ext_ent][key])								
		    def get_ext_ent_class(self, ext_ent: str):								
		        return globals()[self._configuration[ext_ent]['class']]								
		    def get_ext_ent_addr(self, ext_ent: str, key: str):								
		        host = self._configuration[ext_ent][key]['host']								
		        port = int(self._configuration[ext_ent][key]['port'])								
		        return SocketAddress(host, port)								
		    def get_ext_ent_names(self):								
		        return self._configuration['targets']								
										
										
										
		class TestRunConfiguration(unittest.TestCase):								
		    ALT_JSON = '''{"include":["foobar.py"],								
		                   "clock": "MockClock",								
		                   "targets": ["cfs", "foo", "bar"],								
		                   "cfs": {"executable":"CFS_EXE",								
		                           "directory":"CFS_DIR",								
		                           "storage":"../CFS_TMP",								
		                           "class": "CfsExternalEntity"}								
		                  }'''								
		    def test_reading_file(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'))								
		        self.assertTrue(uut._configuration['clock'] == 'StandardClock')								
		    def test_reading_string(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'),								
		                               alt_json=self.ALT_JSON)								
		        self.assertTrue(uut._configuration['include'][0] == 'foobar.py')								
		    def test_file_expansion(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'))								
		        files = uut.files()								
		        self.assertEqual(files[0], os.path.abspath('./example/exampletest1.py'))								
		    def test_cfs_executable(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'),								
		                               alt_json=self.ALT_JSON)								
		        self.assertEqual(uut.get_ext_ent_value('cfs','executable'), 'CFS_EXE')								
		    def test_cfs_directory(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'),								
		                               alt_json=self.ALT_JSON)								
		        self.assertEqual(uut.get_ext_ent_path('cfs','directory'),								
		                         os.path.abspath('./example/CFS_DIR'))								
		    def test_ext_ent_path(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'),								
		                               alt_json=self.ALT_JSON)								
		        self.assertEqual(uut.get_ext_ent_path('cfs', 'directory'),								
		                         os.path.abspath('./example/CFS_DIR'))								
		        self.assertEqual(uut.get_ext_ent_path('cfs', 'storage'), os.path.abspath('CFS_TMP'))								
		    def test_ext_ent_class(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'),								
		                               alt_json=self.ALT_JSON)								
		        self.assertEqual(uut.get_ext_ent_class('cfs'), CfsExternalEntity)								
		    def test_cfs_storage(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'),								
		                               alt_json=self.ALT_JSON)								
		        self.assertEqual(uut.get_ext_ent_path('cfs','storage'), os.path.abspath('CFS_TMP'))								
		    def test_relative_cfs(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'))								
		        self.assertEqual(uut.get_ext_ent_path('cfs','directory'),								
		                         os.path.abspath('./cpu1'))								
		    def test_clock(self) -> None:								
		        uut = RunConfiguration(os.path.abspath('./example'),								
		                               alt_json=self.ALT_JSON)								
		        self.assertEqual(type(MockClock), type(uut.get_clock_class()))								
		    def test_socket_address(self):								
		        CMD = SocketAddress('localhost', 1234)								
		        uut = RunConfiguration(os.path.abspath('./example'))								
		        self.assertEqual(CMD, uut.get_ext_ent_addr('cfs', 'command'))								
		    def test_get_ext_ent_names(self):								
		        uut = RunConfiguration(os.path.abspath('./example'),								
		                               alt_json=self.ALT_JSON)								
		        self.assertEqual(uut.get_ext_ent_names(), ['cfs', 'foo', 'bar'])								
										
										
		################################################################################								
		#TODO: create an abstract base class and turn this implementation into a CcsdsSspViaUdpTransponder								
		class ExternalEntityTransponder(OutputMixin):								
		    '''								
		    Communicate with an entity that is external to the acceptance test.								
		    '''								
		    def __init__(self,								
		                 name: str, clock: StandardClock, config: RunConfiguration) -> None:								
		        self._name = name								
		        self._clock = clock								
		        self._config = config								
		        self._addr_rx = self._config.get_ext_ent_addr(name, 'telemetry')								
		        self._addr_tx = self._config.get_ext_ent_addr(name, 'command')								
		        self._queue = deque()								
		        self._receiver = PacketReceiver(self._queue, self._clock, self._addr_rx)								
		        self._thread = threading.Thread(target=self._receiver.run)								
		        self._transmitter = PacketSender(self._addr_tx)								
		        self._ctdb = CtDatabase(self._config.get_ext_ent_path(name, 'database'))								
		        self._archive = PacketArchive(name, self._ctdb, self._queue)								
		        self.Applications = self._ctdb.Applications								
		        self.Telemetry = self._ctdb.Telemetry								
		        self.CommandCodes = self._ctdb.CommandCodes								
		    def start(self) -> None:								
		        self._thread.start()								
		        self._transmitter.start()								
		    def stop(self) -> None:								
		        self._receiver.signal_shutdown()								
		        self._transmitter.stop()								
		        self._thread.join()								
		    def process(self) -> None:								
		        self._archive.process()								
		    def has_sample(self, id: SampleId) -> bool:								
		        return self._archive.has_sample(MagicMid(type_code=id))								
		    def count_sample(self, id: SampleId) -> int:								
		        return self._archive.count_sample(MagicMid(type_code=id))								
		    def total_sample_count(self) -> int:								
		        return self._archive.total_count()								
		    def get_last_sample(self, id: SampleId) -> PacketSample:								
		        return self._archive.get_last_sample(MagicMid(type_code=id))								
		    def decom_sample(self, id: SampleId, idx: int = -1):								
		        return self._archive.decom_sample(MagicMid(type_code=id), idx)								
		    def send_bytes(self, datagram: bytes) -> None:								
		        self._transmitter.send_bytes(datagram)								
		    def send_command(self, cmd: ParameterlessCommandPacket) -> None:								
		        self._transmitter.send_bytes(cmd.to_bytes())								
		    def command(self, cmd) -> CommandBuilder:								
		        warning_text = "The method command(cmd) is deprecated. Please use get_command_builder(cmd)."								
		        warnings.warn(warning_text, DeprecationWarning, stacklevel=2)								
		        print(warning_text)								
		        self.print_call_location(stack_offset=2, force_print=True)								
		        return self.get_command_builder(cmd)								
		    def get_command_builder(self, cmd) -> CommandBuilder:								
		        builder = CommandBuilder(self._ctdb, cmd)								
		        builder.set_transponder(self)								
		        return builder								
		    def get_telemetry_tracker(self, tlm) -> PacketTracker:								
		        tracker = PacketTracker(self._clock, self._archive, tlm)								
		        return tracker								
										
										
		class TestExternalEntityTransponder(unittest.TestCase):								
		    ALT_JSON = '''{"include":["foobar.py"],								
		                   "clock": "StandardClock",								
		                   "targets": ["cfs", "foo", "bar"],								
		                   "cfs": {"executable":"CFS_EXE",								
		                           "directory":"CFS_DIR",								
		                           "storage":"../CFS_TMP",								
		                           "class": "CfsExternalEntity",								
		                           "database": "../ct.json",								
		                           "command": {"host": "localhost", "port": 6565},								
		                           "telemetry": {"host": "localhost", "port": 6565}								
		                          }								
		                  }'''								
		    CFE_ES_CMD_MID = SampleId(0x1806)								
		    PACKET_1 = ParameterlessCommandPacket(CFE_ES_CMD_MID, 1, 2)								
		    PACKET_2 = ParameterlessCommandPacket(CFE_ES_CMD_MID, 2, 3)								
		    def setUp(self) -> None:								
		        clock = StandardClock()								
		        self._config = RunConfiguration(os.path.abspath('./example'),								
		                               alt_json=self.ALT_JSON)								
		        self.uut = ExternalEntityTransponder('cfs', clock, self._config)								
		    def test_ctdb(self) -> None:								
		        self.assertEqual(self.uut._ctdb.Telemetry.CFE_TIME_DIAG_TLM.name, 'CFE_TIME_DIAG_TLM')								
		    def test_basic_conops(self) -> None:								
		        # delay checks until after stop() is called								
		        self.uut.start()								
		        has_sample_check_1 = self.uut.has_sample(self.CFE_ES_CMD_MID)								
		        self.uut.send_bytes(self.PACKET_1.to_bytes())								
		        time.sleep(0.1)								
		        time.sleep(0.1)								
		        time.sleep(0.1)								
		        has_sample_check_2 = self.uut.has_sample(self.CFE_ES_CMD_MID)								
		        self.uut.send_command(self.PACKET_2)								
		        time.sleep(0.2)								
		        sample_count_check = self.uut.count_sample(self.CFE_ES_CMD_MID)								
		        sample = self.uut.get_last_sample(self.CFE_ES_CMD_MID)								
		        self.uut.stop()								
		        self.assertFalse(has_sample_check_1)								
		        self.assertTrue(has_sample_check_2) # sometimes fails :-(								
		        self.assertEqual(2, sample_count_check)								
		        self.assertEqual(sample.datagram, self.PACKET_2.to_bytes())								
		    def test_command_codes_enum(self) -> None:								
		        self.assertEqual(0, # the no-op commands are always command code 0								
		                         self.uut.CommandCodes.TO_NOP_CC.cc())								
										
										
		################################################################################								
		class ExternalEntityRunner(OutputMixin):								
		    '''								
		    ExternalEntityRunner is a runner base class that does nothing.								
		    Override to create real runners.								
		    '''								
		    _running = False								
		    _start_count = 0								
		    _stop_count = 0								
		    def start(self) -> None:								
		        self._running = True								
		        self._start_count += 1								
		    def stop(self) -> None:								
		        self._running = False								
		        self._stop_count += 1								
		    def isRogue(self) -> bool:								
		        return False								
		    def isRunning(self) -> bool:								
		        return self._running								
		    def cleanUp(self) -> None:								
		        pass								
										
										
		################################################################################								
		# need to account for cFS temporary/persistent files								
		class CfsRunner(ExternalEntityRunner):								
		    def __init__(self, name: str, cfg: RunConfiguration) -> None:								
		        self._name = name								
		        self.exe = cfg.get_ext_ent_value(name,'executable')								
		        self.where = cfg.get_ext_ent_path(name,'directory')								
		        self.full_path = os.path.abspath(PurePath(self.where, self.exe))								
		        self.storage = cfg.get_ext_ent_path(name,'storage')								
		        self.process = None								
		    def start(self) -> None:								
		        super().start()								
		        self.print_verbose(f'****cFS STARTING "{self._name}": {self.full_path} in {self.storage}')								
		        self.process = subprocess.Popen([self.full_path],								
		                                        stdout=subprocess.PIPE,								
		                                        stderr=subprocess.PIPE,								
		                                        universal_newlines=True,								
		                                        cwd=self.storage)								
		        self._wait_for_listening()								
		    def stop(self) -> None:								
		        super().stop()								
		        self.process.terminate()								
		        self.process.wait()								
		        self.print_verbose(f'****cFS STOPPED "{self._name}"')								
		        for line in self.process.stdout:								
		            self.print_verbose(line.rstrip())								
		        self._close()								
		    def isRogue(self) -> bool:								
		        result = self._pgrep()								
		        return result.returncode == 0								
		    def isRunning(self) -> bool:								
		        result = self.process.poll()								
		        return result is None								
		    def cleanUp(self) -> None:								
		        result = self._pgrep()								
		        if result.returncode == 0:								
		            pid = int(result.stdout)								
		            os.kill(pid, signal.SIGTERM)								
		            status = os.waitpid(pid, 0)								
		    def _close(self) -> None:								
		        self.process.stdout.close()								
		        self.process.stderr.close()								
		    def _pgrep(self) -> subprocess.CompletedProcess:								
		        # stdout=PIPE, stderr=PIPE								
		        #return subprocess.run(["pgrep", self.exe], capture_output=True)								
		        return subprocess.run(["pgrep", self.exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE)								
		    def _is_listening(self) -> bool:								
		        #result = subprocess.run(["netstat", "-ulpn"],								
		        #                        capture_output=True, text=True)								
		        result = subprocess.run(["netstat", "-ulpn"],								
		                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,								
		                                encoding=locale.getpreferredencoding(False))								
		        return result.stdout.find(self.exe) >= 0								
		    def _wait_for_listening(self) -> None:								
		        while not self._is_listening():								
		            pass								
		        self.print_verbose(f'****cFS LISTENING "{self._name}"')								
										
										
		class TestCfsRunner(unittest.TestCase):								
		    ALT_JSON = '''{"include":["foobar.py"],								
		                   "clock": "MockClock",								
		                   "cfs": {"executable":"core-cpu1",								
		                           "directory":"cpu1",								
		                           "storage":"cpu1"}								
		                  }'''								
		    def setUp(self) -> None:								
		        self.storage = os.path.abspath(PurePath(Path.cwd(), 'cpu1'))								
		        self.cfg = RunConfiguration(os.path.abspath('.'),								
		                               alt_json=self.ALT_JSON)								
		        self.uut = CfsRunner('cfs', self.cfg)								
		    def test_isRogue_false(self) -> None:								
		        self.assertFalse(self.uut.isRogue())								
		    def test_isRogue_true(self) -> None:								
		        self.uut.silence()								
		        self.uut.start()								
		        self.assertTrue(self.uut.isRogue())								
		        self.uut.stop()								
		    def test_isRunning(self) -> None:								
		        self.uut.silence()								
		        self.uut.start()								
		        self.assertTrue(self.uut.isRunning())								
		        self.uut.stop()								
		        self.assertFalse(self.uut.isRunning())								
		    def test_cleanUp(self) -> None:								
		        cfsPath = os.path.abspath(PurePath(self.storage, 'core-cpu1'))								
		        process = subprocess.Popen([cfsPath],								
		            cwd=self.storage,								
		            stdout=subprocess.PIPE,								
		            stderr=subprocess.PIPE,								
		            universal_newlines=True)								
		        self.uut.cleanUp()								
		        self.assertFalse(self.uut.isRogue())								
		        process.wait()								
		        process.stdout.close()								
		        process.stderr.close()								
										
										
		################################################################################								
		FailureInstance = namedtuple("FailureInstance", "id msg")								
										
		class AcceptanceTestRecorder(OutputMixin):								
		    STAND_OUT = '######'								
		    def __init__(self, output_level: OutputLevel = OutputLevel.NORMAL) -> None:								
		        self._test_count = 0								
		        self._current_test = ''								
		        self._failures = []								
		    def report(self) -> None:								
		        self.print_normal(self._report())								
		    def _report(self) -> str:								
		        if self.fail_count() == 0:								
		            return f'{self.test_count()} tests ran, all passed!'								
		        else:								
		            return f'{self.test_count()} tests ran, {self.fail_count()} failed!'								
		    def start_test(self, id: str) -> None:								
		        self._test_count += 1								
		        self._current_test = id								
		        #print(f'(starting test {id})')								
		    def fail(self, reason: str, stack_adjust:int = 0) -> None:								
		        msg = f'{self.STAND_OUT} Test {self._current_test} ' \								
		            f'failed for reason "{reason}"'								
		        self.print_normal(msg)								
		        self.print_call_location(stack_offset=3+stack_adjust, prefix=f'{self.STAND_OUT} ')								
		        self._failures.append(FailureInstance(self._current_test, reason) )								
		    def fail_exception(self, reason: str, alt_tb) -> None:								
		        msg = f'{self.STAND_OUT} Test {self._current_test} ' \								
		            f'failed for reason "{reason}"'								
		        self.print_normal(msg)								
		        self.print_call_location(alt_tb=alt_tb, prefix=f'{self.STAND_OUT} ')								
		        self._failures.append(FailureInstance(self._current_test, reason) )								
		    def test_count(self) -> int:								
		        return self._test_count								
		    def fail_count(self) -> int:								
		        return len(self._failures)								
		    def have_failures(self) -> bool:								
		        return self.fail_count() > 0								
		    def info(self, msg: str) -> None:								
		        self.print_normal(msg)								
										
		class TestAcceptanceTestRecorder(unittest.TestCase):								
		    def test_one(self) -> None:								
		        uut = AcceptanceTestRecorder()								
		        uut.silence()								
		        uut.start_test('one')								
		        uut.fail('just because')								
		        self.assertEqual(uut._test_count, 1)								
		        self.assertEqual(len(uut._failures), 1)								
		        self.assertEqual(uut._failures[0].id, 'one')								
		        self.assertEqual(uut._failures[0].msg, 'just because')								
										
										
		################################################################################								
		class ExternalEntity(OutputMixin):								
		    '''								
		    An external entity is basically a service that the acceptance test runner								
		    uses to help execute a test. The primary example is a cFS entity that								
		    starts up a cFS process, controls the lifetime, and provides for communication								
		    with the entity. This base class provides the basic lifecycle methods (via the								
		    external entity runner) as well as a method to access the external entity								
		    transponder.								
		    '''								
		    _runner = None								
		    _transponder = None								
		    def __init__(self, name: str, clock: StandardClock, config=None) -> None:								
		        self._name = name								
		        self._clock = clock								
		    def start(self) -> None:								
		        self._runner.start()								
		        self._transponder.start()								
		    def stop(self) -> None:								
		        self._transponder.stop()								
		        self._runner.stop()								
		    def get_runner(self) -> ExternalEntityRunner:								
		        return self._runner								
		    def get_transponder(self) -> ExternalEntityTransponder:								
		        return self._transponder								
		    def get_name(self) -> str:								
		        return self._name								
										
										
		################################################################################								
		class CfsExternalEntity(ExternalEntity):								
		    def __init__(self, name: str, clock: StandardClock, config: RunConfiguration) -> None:								
		        super().__init__(name, clock, config)								
		        self._transponder = ExternalEntityTransponder(								
		            name = name,								
		            clock = self._clock,								
		            config = config)								
		        self._runner = CfsRunner(name, config)								
										
										
		################################################################################								
		#TODO(djk): need a better name!								
		class AcceptanceObjectSource(OutputMixin):								
		    def __init__(self,								
		                 config=RunConfiguration) -> None:								
		        self._config = config								
		        self._atr = AcceptanceTestRecorder()								
		        self._external_entities = {}								
		        # pick/set clock based on config								
		        #TODO(djk): clocks might have init param (to set mission time)								
		        self._clock = config.get_clock_class()()								
		    def config(self) -> RunConfiguration:								
		        return self._config								
		    def recorder(self) -> AcceptanceTestRecorder:								
		        return self._atr								
		    def clock(self) -> StandardClock:								
		        return self._clock								
		    def new_ext_ent(self, ext_ent_name: str) -> None:								
		        # External entities have threads which can only be run once.								
		        # Need to be able to "restart" so need to delete/recreate.								
		        if ext_ent_name in self._external_entities:								
		            del self._external_entities[ext_ent_name]								
		        ext_ent_class = self._config.get_ext_ent_class(ext_ent_name)								
		        ext_ent = ext_ent_class(ext_ent_name, self._clock, self._config)								
		        ext_ent.set_output_level(self.output_level)								
		        self._external_entities[ext_ent_name] = ext_ent								
		    def get_ext_ent(self, ext_ent_name: str) -> ExternalEntity:								
		        if ext_ent_name not in self._external_entities:								
		            self.new_ext_ent(ext_ent_name)								
		        return self._external_entities[ext_ent_name]								
		    def set_output_level(self, output_level: OutputLevel) -> None:								
		        super().set_output_level(output_level)								
		        for e in self._external_entities:								
		            e.set_output_level(output_level)								
										
		################################################################################								
		# An AcceptanceSuite child class represents a set of tests (define								
		# "test*" methods for individual tests) where cFS is started once for the								
		# suite and shutdown when the suite is done.								
		class AcceptanceSuite:								
		    '''								
		    This is the base class for cFS acceptance tests that will be found, loaded,								
		    and run by the acceptanceRunner.py script.								
										
		    Since starting and stopping cFS takes time, suites are used to bundle								
		    ordered groups of tests that will run relative to a single invocation								
		    of cFS.								
										
		    When a test suite begins, setUpSuite is called once. Before each individual								
		    test is run, setUpTest is called. Tests are run in the order they appear								
		    in the source code (suites are run in alphabetical order). After each								
		    test is run, tearDownTest is called, and after the suite completes,								
		    tearDownSuite is called.								
										
		    Test methods names must begin with "test", take only self, and return None.								
										
		    Within test methods, the following assertions are available:								
		        assertTrue(bool)								
		        assertFalse(bool)								
		        assertEqual(a,b)								
		        assertNotEqual(a,b)								
										
		    Additionally, self.fail(msg) may be called.								
		    '''								
		    def __init__(self, id: str, source: AcceptanceObjectSource) -> None:								
		        self._source = source								
		        self._id = id								
		        self._test_names = self._collect_tests()								
		    def _collect_tests(self) -> Sequence[str]:								
		        method_names = [name for name in dir(self) if \								
		                        inspect.ismethod(getattr(self, name))]								
		        test_names = [test_name for test_name in method_names \								
		                      if test_name.startswith('test')]								
		        line_numbers = [getattr(self, m).__code__.co_firstlineno for m in \								
		                        test_names]								
		        ordered_tests = [t for _, t in sorted(zip(line_numbers, test_names), \								
		                                              key=lambda pair: pair[0])]								
		        return ordered_tests								
		    def name(self) -> str:								
		        return self._id								
		    def run(self) -> None:								
		        self._start_suite()								
		        self._run_tests()								
		        self._stop_suite()								
		    def _start_suite(self) -> None:								
		        self._source.recorder().info(f'Starting suite {self._id}')								
		        for ext_ent_name in self._source.config().get_ext_ent_names():								
		            self._source.new_ext_ent(ext_ent_name)								
		            self._source.get_ext_ent(ext_ent_name).start()								
		        self.setUpSuite()								
		    def _stop_suite(self) -> None:								
		        self.tearDownSuite()								
		        for ext_ent_name in reversed(self._source.config().get_ext_ent_names()):								
		            self._source.get_ext_ent(ext_ent_name).stop()								
		    def _run_tests(self) -> None:								
		        for test_name in self._test_names:								
		            self._run_test(test_name)								
		    def _run_test(self, test_name: str) -> None:								
		        self._start_test(test_name)								
		        self._execute_test(test_name)								
		        self._stop_test(test_name)								
		    def _start_test(self, test_name: str) -> None:								
		        self._current_test = f'{self._id}.{test_name}'								
		        self._source.recorder().start_test(self._current_test)								
		        self._source.recorder().info(f' Starting test {self._current_test}')								
		        self.setUpTest()								
		    def _stop_test(self, test_name: str) -> None:								
		        self.tearDownTest()								
		    def _execute_test(self, test_name: str) -> None:								
		        test_method = getattr(self, test_name)								
		        try:								
		            test_method()								
		        except BaseException as e:								
		            msg = f'Fail {self._current_test}: test ' \								
		                f'raised an exception: {e}, {type(e)}'								
		            exception_source_information = traceback.format_tb(sys.exc_info()[-1])[1]								
		            self._source.recorder().fail_exception(msg, exception_source_information)								
										
		    def setUpSuite(self) -> None:								
		        '''setUpSuite is called once before a suite of tests begins.'''								
		        pass								
		    def tearDownSuite(self) -> None:								
		        '''tearDownSuite is called once after a suite of tests ends.'''								
		        pass								
										
		    def setUpTest(self) -> None:								
		        '''setUpTest is called before each test begins.'''								
		        pass								
		    def tearDownTest(self) -> None:								
		        '''tearDownTest is called after each test ends.'''								
		        pass								
										
		    def assertTrue(self, v: bool, msg: str='') -> None:								
		        if not v:								
		            self._source.recorder().fail \								
		                (f'Fail {self._current_test}: assertTrue passed False. {msg}')								
		    def assertFalse(self, v: bool, msg: str='') -> None:								
		        if v:								
		            self._source.recorder().fail \								
		                (f'Fail {self._current_test}: assertFalse passed True. {msg}')								
		    def assertEqual(self, a, b, msg: str='') -> None:								
		        if a != b:								
		            self._source.recorder().fail \								
		                (f'Fail {self._current_test}: assertEqual {a} != {b}. {msg}')								
		    def assertNotEqual(self, a, b, msg: str='') -> None:								
		        if a == b:								
		            self._source.recorder().fail \								
		                (f'Fail {self._current_test}: assertNotEqual {a} == {b}. {msg}')								
		    def fail(self, msg: str='') -> None:								
		        self._source.recorder().fail \								
		            (f'Fail {self._current_test}: explicit fail. {msg}')								
		    def print(self, *args, **kwargs) -> None:								
		        if self._source.output_level != OutputLevel.SILENT:								
		            print(*args, **kwargs)								
		    def get_transponder(self, name: str) -> ExternalEntityTransponder:								
		        # will use this to look up from pool (cfs, sim, etc.)								
		        return self._source.get_ext_ent(name).get_transponder()								
		    def get_clock(self) -> StandardClock:								
		        return self._source.get_clock()								
		    def get_time(self) -> Nanoseconds:								
		        return self._source._clock.get_time()								
		    def wait(self, s: Seconds) -> None:								
		        self._source._clock.wait(s)								
										
										
		################################################################################								
		class ATestDriverSuite(AcceptanceSuite):								
		    def test_equal_pass(self) -> None:								
		        self.print('this should not be seen because the test is silent')								
		        self.assertEqual(1+1, 2)								
		    def test_equal_fail(self) -> None:								
		        self.assertEqual(1+1, 3)								
		    def test_true_pass(self) -> None:								
		        self.assertTrue(True)								
		    def test_true_fail(self) -> None:								
		        self.assertTrue(False)								
		    def test_false_pass(self) -> None:								
		        self.assertFalse(False)								
		    def test_false_fail(self) -> None:								
		        self.assertFalse(True)								
		    def test_not_equal_pass(self) -> None:								
		        self.assertNotEqual(1+1, 3)								
		    def test_not_equal_fail(self) -> None:								
		        self.assertNotEqual(1+1, 2)								
		    def test_fail(self) -> None:								
		        self.fail()								
		    def test_exception_failure(self) -> None:								
		        raise NameError("exception failure")								
		    def test_send_a_command(self) -> None:								
		        cfs = self.get_transponder('cfs')								
		        cmd = cfs.get_command_builder(cfs.CommandCodes.TO_OUTPUT_ENABLE_CC)								
										
		class MockCfsEntity(ExternalEntity):								
		    def __init__(self, name: str, clock: StandardClock, config: RunConfiguration) -> None:								
		        super().__init__(name, clock)								
		        self._runner = ExternalEntityRunner()								
		        self._transponder = ExternalEntityTransponder(name, clock, config)								
										
		class TestAcceptanceSuite(unittest.TestCase):								
		    ALT_JSON = '''{"include":["foobar.py"],								
		                   "clock": "StandardClock",								
		                   "targets": ["cfs"],								
		                   "cfs": {"class": "MockCfsEntity",								
		                           "executable":"CFS_EXE",								
		                           "directory":"CFS_DIR",								
		                           "storage":"CFS_TMP",								
		                           "database": "../ct.json",								
		                           "command": {"host": "localhost", "port": 6565},								
		                           "telemetry": {"host": "localhost", "port": 6565}								
		                          }								
		                  }'''								
		    NUMBER_OF_MOCK_TESTS = 11								
		    NUMBER_OF_FAILING_TESTS = 6								
		    def setUp(self) -> None:								
		        self.config = RunConfiguration('./example', alt_json=self.ALT_JSON)								
		        self.source = AcceptanceObjectSource(self.config)								
		        self.source.silence()								
		        self.uut = ATestDriverSuite('unit_test.ATestDriverSuite', self.source)								
		    def test_collect_test_names(self) -> None:								
		        self.assertEqual(len(self.uut._test_names), self.NUMBER_OF_MOCK_TESTS)								
		    def test_run_count_tests(self) -> None:								
		        self.uut.run()								
		        self.assertEqual(self.NUMBER_OF_MOCK_TESTS,								
		                         self.source.recorder().test_count())								
		    def test_run_count_fails(self) -> None:								
		        self.uut.run()								
		        self.assertEqual(self.NUMBER_OF_FAILING_TESTS,								
		                         self.source.recorder().fail_count())								
		    def test_run_count_start(self) -> None:								
		        self.assertEqual(0, self.source.get_ext_ent('cfs')._runner._start_count)								
		        self.uut.run()								
		        self.assertEqual(1, self.source.get_ext_ent('cfs')._runner._start_count)								
		    def test_run_count_stop(self) -> None:								
		        self.assertEqual(0, self.source.get_ext_ent('cfs')._runner._stop_count)								
		        self.uut.run()								
		        self.assertEqual(1, self.source.get_ext_ent('cfs')._runner._stop_count)								
		    def test_cfs_running(self) -> None:								
		        self.assertFalse(self.source.get_ext_ent('cfs')._runner._running)								
		        self.uut.run()								
		        self.assertFalse(self.source.get_ext_ent('cfs')._runner._running)								
										
										
		################################################################################								
		class AcceptanceModule:								
		    '''								
		    Tests are in suites, suites are in modules (Python files). This takes the								
		    module (file) and the list of suites (classes) and creates test suite								
		    objects.								
		    '''								
		    def __init__(self,								
		                 module_name: str,								
		                 file_path: str,								
		                 suites: Sequence[Type[AcceptanceSuite]]) -> None:								
		        self.module_name = module_name								
		        self.file_path = file_path								
		        self.suites = suites								
		    def make_suites(self,								
		                    source: AcceptanceObjectSource) -> \								
		                    Sequence[Type[AcceptanceSuite]]:								
		        return [c(f'{self.module_name}.{c.__name__}', source) for \								
		                c in self.suites]								
										
		################################################################################								
		class AcceptanceRunner(OutputMixin):								
		    def __init__(self, config=RunConfiguration) -> None:								
		        #TODO(djk): check validity of config								
		        self._source = AcceptanceObjectSource(config=config)								
		    def load(self) -> None:								
		        '''Collect all of the acceptance test suites.'''								
		        self._suites = self._test_suites()								
		    def run(self) -> None:								
		        '''Run all of the acceptance test suites.'''								
		        for suite in self._suites:								
		            suite.run()								
		    def report(self) -> None:								
		        '''Report on all of the acceptance test suites.'''								
		        self._source.recorder().report()								
		    def have_failures(self) -> bool:								
		        return self._source.recorder().have_failures()								
		    def _files(self):								
		        return [(os.path.basename(p)[:-3], p) for p in \								
		                self._source.config().files()]								
		    def _test_modules(self) -> Sequence[AcceptanceModule]:								
		        answer = []								
		        for module_name, file_path in self._files():								
		            spec = importlib.util.spec_from_file_location(module_name,								
		                                                          file_path)								
		            module = importlib.util.module_from_spec(spec)								
		            spec.loader.exec_module(module)								
		            class_names = [name for name in dir(module) if \								
		                           inspect.isclass(inspect.getattr_static(module,								
		                                                                  name))]								
		            classes = [inspect.getattr_static(module, name) for \								
		                       name in class_names]								
		            suites = [c for c in classes if self._is_suite(c)]								
		            if len(suites) > 0:								
		                answer.append(AcceptanceModule(module_name, file_path, suites))								
		        return answer								
		    def _is_suite(self, cls) -> bool:								
		        return any(map(lambda bc: \								
		                       bc.__name__ == 'AcceptanceSuite', cls.__bases__))								
		    def _test_suites(self) -> Sequence[Type[AcceptanceSuite]]:								
		        answer = []								
		        for modual in self._test_modules():								
		            answer += modual.make_suites(self._source)								
		        return answer								
										
		class TestAcceptanceRunner(unittest.TestCase):								
		    def setUp(self) -> None:								
		        self._config = RunConfiguration(os.path.abspath('./example'))								
		        self._uut = AcceptanceRunner(config=self._config)								
		        self._uut.silence()								
		    def test_include_files(self) -> None:								
		        files = self._uut._files()								
		        self.assertEqual(files[0][0], 'exampletest1')								
		        self.assertEqual(files[0][1],								
		                         os.path.abspath('./example/exampletest1.py'))								
		    def test_include_test_modules(self) -> None:								
		        modules = self._uut._test_modules()								
		        self.assertEqual(modules[0].suites[0].__name__, 'MySuite')								
		    def test_include_test_suites(self) -> None:								
		        suites = self._uut._test_suites()								
		        self.assertEqual(suites[0].name(), 'exampletest1.MySuite')								
										
										
										
		################################################################################								
		def main() -> None:								
		    parser = argparse.ArgumentParser()								
		    parser.add_argument("-vv", help="very verbose", action="store_true")								
		    parser.add_argument("-v", help="verbose", action="store_true")								
		    parser.add_argument("-s", help="silent", action="store_true")								
		    args = parser.parse_args()								
		    output_level = OutputLevel.NORMAL								
		    if args.vv:								
		        output_level = OutputLevel.SPAM								
		    if args.v:								
		        output_level = OutputLevel.VERBOSE								
		    if args.s:								
		        output_level = OutputLevel.SILENT								
		    config = RunConfiguration() #TODO(djk): catch error and give nice output								
		    runner = AcceptanceRunner(config)								
		    runner.set_output_level(output_level)								
		    runner.load()								
		    runner.run()								
		    runner.report()								
		    if runner.have_failures():								
		       sys.exit(1)								
										
		################################################################################								
		if __name__ == "__main__":								
		    main()								
		