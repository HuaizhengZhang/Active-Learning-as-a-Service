#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 23, 2022
"""
from typing import Generic, TypeVar

from pydantic import validator
from pydantic.generics import GenericModel

RequiredType = TypeVar('RequiredType')


class TypeCheckMixin(GenericModel, Generic[RequiredType]):
    """For auto detecting configuration class by set value of :code:`type`.
    """
    type: RequiredType

    __required_type__: RequiredType

    @validator('type')
    def check_type(cls, required_type: RequiredType) -> RequiredType:
        """
        Checks type value provided is the same as the required value.
        This is to generate validator for check :code:`type` field of subclasses of Generic typpe :class:`RequiredType`.
        """
        if required_type != cls.__required_type__:
            raise ValueError(f'Expected {cls.__required_type__} but got {required_type}')
        return required_type
