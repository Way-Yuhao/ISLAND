from abc import ABC, abstractmethod


class MyAbstractBaseClass(ABC):

    def __init__(self):
        self.a = 0

    @abstractmethod
    def my_abstract_method(self):
        pass

    def alert(self):
        print("This is an alert method.")


# This class must implement my_abstract_method to be instantiable.
class ConcreteClass(MyAbstractBaseClass):

    def __init__(self):
        # super().__init__()
        self.b = 1

    def my_abstract_method(self):
        print("Implementation of the abstract method.")


# Attempting to instantiate MyAbstractBaseClass would raise an error.
# concrete_instance = MyAbstractBaseClass()  # This would raise an error.

# Correctly implemented subclass can be instantiated.
concrete_instance = ConcreteClass()
concrete_instance.my_abstract_method()
concrete_instance.alert()
print(concrete_instance.a)
print(concrete_instance.b)
