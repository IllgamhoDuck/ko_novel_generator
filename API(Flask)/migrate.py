from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
import app, migrate

migrate = Migrate(app, migrate)

manager = Manager(app)
manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()