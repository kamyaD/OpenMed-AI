from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from accounts.models import Role, UserProfile
from inventory.models import Facility

DEMO = [
    ("admin1", Role.ADMIN, True),
    ("analyst1", Role.ANALYST, False),
    ("manager1", Role.SUPPLY_MANAGER, False),
    ("facility1", Role.FACILITY_USER, False),
]


class Command(BaseCommand):
    help = "Create demo users, one per role (development only)"

    def add_arguments(self, parser):
        parser.add_argument("--password", default="ChangeMe123!")

    def handle(self, *args, **opts):
        User = get_user_model()
        for username, role, staff in DEMO:
            user, _ = User.objects.get_or_create(username=username)
            user.set_password(opts["password"])
            user.is_staff = staff
            user.save()
            profile, _ = UserProfile.objects.update_or_create(user=user, defaults={"role": role})
            if role == Role.FACILITY_USER:
                profile.facilities.set(Facility.objects.all()[:2])
            self.stdout.write(f"{username} -> {role}")