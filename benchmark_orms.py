# benchmark_orms.py
"""
Benchmark suite for comparing Ormax ORM with other popular ORMs
"""
import asyncio
import time
import statistics
from typing import List
import os
from datetime import datetime

# Ormax imports
try:
    from ormax import Database as OrmaxDatabase, Model as OrmaxModel
    from ormax.fields import *
    ORMAX_AVAILABLE = True
except ImportError:
    ORMAX_AVAILABLE = False
    print("Ormax not available")

# SQLAlchemy imports
try:
    from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
    from sqlalchemy.orm import sessionmaker, declarative_base
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("SQLAlchemy not available")

# Tortoise ORM imports
try:
    from tortoise.models import Model as TortoiseModel
    from tortoise import fields as tortoise_fields
    TORTOISE_AVAILABLE = True
except ImportError:
    TORTOISE_AVAILABLE = False
    print("Tortoise ORM not available")

# Peewee imports
try:
    import peewee
    from playhouse.sqlite_ext import SqliteExtDatabase
    PEWEE_AVAILABLE = True
except ImportError:
    PEWEE_AVAILABLE = False
    print("Peewee not available")


class BenchmarkResult:
    """Store benchmark results"""
    def __init__(self, orm_name: str):
        self.orm_name = orm_name
        self.results = {}
    
    def add_result(self, operation: str, times: List[float]):
        self.results[operation] = {
            'times': times,
            'total': sum(times),
            'average': statistics.mean(times) if times else 0,
            'median': statistics.median(times) if len(times) > 1 else 0,
            'min': min(times) if times else 0,
            'max': max(times) if times else 0
        }
    
    def __str__(self):
        output = f"\n🏁 {self.orm_name} Benchmark Results:\n"
        output += "=" * 50 + "\n"
        for operation, data in self.results.items():
            output += f"{operation:15} | Avg: {data['average']:.4f}s | Total: {data['total']:.4f}s\n"
        output += "=" * 50 + "\n"
        return output


# Ormax Models
if ORMAX_AVAILABLE:
    class OrmaxUser(OrmaxModel):
        table_name = "ormax_users"
        id = AutoField()
        username = CharField(max_length=50, unique=True)
        email = EmailField(unique=True)
        is_active = BooleanField(default=True)
        created_at = DateTimeField(auto_now_add=True)

    class OrmaxPost(OrmaxModel):
        table_name = "ormax_posts"
        id = AutoField()
        title = CharField(max_length=200)
        content = TextField()
        author_id = IntegerField(foreign_key='ormax_users.id')
        published = BooleanField(default=False)
        created_at = DateTimeField(auto_now_add=True)


# SQLAlchemy Models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class SQLAlchemyUser(Base):
        __tablename__ = 'sqlalchemy_users'
        id = Column(Integer, primary_key=True)
        username = Column(String(50), unique=True)
        email = Column(String(100), unique=True)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.now)
    
    class SQLAlchemyPost(Base):
        __tablename__ = 'sqlalchemy_posts'
        id = Column(Integer, primary_key=True)
        title = Column(String(200))
        content = Column(Text)
        author_id = Column(Integer)
        published = Column(Boolean, default=False)
        created_at = Column(DateTime, default=datetime.now)


# Tortoise Models
if TORTOISE_AVAILABLE:
    class TortoiseUser(TortoiseModel):
        id = tortoise_fields.IntField(pk=True)
        username = tortoise_fields.CharField(50, unique=True)
        email = tortoise_fields.CharField(100, unique=True)
        is_active = tortoise_fields.BooleanField(default=True)
        created_at = tortoise_fields.DatetimeField(auto_now_add=True)
        
        class Meta:
            table = "tortoise_users"
    
    class TortoisePost(TortoiseModel):
        id = tortoise_fields.IntField(pk=True)
        title = tortoise_fields.CharField(200)
        content = tortoise_fields.TextField()
        author = tortoise_fields.ForeignKeyField('models.TortoiseUser', related_name='posts')
        published = tortoise_fields.BooleanField(default=False)
        created_at = tortoise_fields.DatetimeField(auto_now_add=True)
        
        class Meta:
            table = "tortoise_posts"


# Peewee Models - رفع مشکل فیلد created_at
if PEWEE_AVAILABLE:
    peewee_db = SqliteExtDatabase('peewee_benchmark.db')
    
    class PeeweeUser(peewee.Model):
        username = peewee.CharField(unique=True, max_length=50)
        email = peewee.CharField(unique=True, max_length=100)
        is_active = peewee.BooleanField(default=True)
        created_at = peewee.DateTimeField(default=datetime.now)  # اضافه کردن مقدار پیش‌فرض
        
        class Meta:
            database = peewee_db
            table_name = 'peewee_users'
    
    class PeeweePost(peewee.Model):
        title = peewee.CharField(max_length=200)
        content = peewee.TextField()
        author = peewee.ForeignKeyField(PeeweeUser, backref='posts')
        published = peewee.BooleanField(default=False)
        created_at = peewee.DateTimeField(default=datetime.now)  # اضافه کردن مقدار پیش‌فرض
        
        class Meta:
            database = peewee_db
            table_name = 'peewee_posts'


class ORMBenchmark:
    """ORM Benchmark Suite"""
    
    def __init__(self, iterations: int = 1000):
        self.iterations = iterations
        self.results = []
    
    async def benchmark_ormax(self) -> BenchmarkResult:
        """Benchmark Ormax ORM"""
        if not ORMAX_AVAILABLE:
            print("Ormax not available, skipping benchmark")
            return None
            
        result = BenchmarkResult("Ormax")
        
        # Setup
        db = OrmaxDatabase("sqlite:///ormax_benchmark.db")
        await db.connect()
        db.register_model(OrmaxUser)
        db.register_model(OrmaxPost)
        await db.drop_tables()
        await db.create_tables()
        
        # Benchmark Create
        create_times = []
        for i in range(self.iterations):
            start_time = time.time()
            user = await OrmaxUser.create(
                username=f"user_{i}",
                email=f"user_{i}@example.com",
                is_active=True
            )
            create_times.append(time.time() - start_time)
        
        result.add_result("Create", create_times)
        
        # Benchmark Read
        read_times = []
        for i in range(100):  # Read 100 times
            start_time = time.time()
            users = await OrmaxUser.objects().all()
            read_times.append(time.time() - start_time)
        
        result.add_result("Read", read_times)
        
        # Benchmark Update
        update_times = []
        for i in range(100):
            start_time = time.time()
            await OrmaxUser.objects().filter(username=f"user_{i}").update(is_active=False)
            update_times.append(time.time() - start_time)
        
        result.add_result("Update", update_times)
        
        # Benchmark Delete
        delete_times = []
        for i in range(100):
            start_time = time.time()
            await OrmaxUser.objects().filter(username=f"user_{i}").delete()
            delete_times.append(time.time() - start_time)
        
        result.add_result("Delete", delete_times)
        
        await db.disconnect()
        return result
    
    def benchmark_sqlalchemy_sync(self) -> BenchmarkResult:
        """Benchmark SQLAlchemy (sync)"""
        if not SQLALCHEMY_AVAILABLE:
            print("SQLAlchemy not available, skipping benchmark")
            return None
            
        result = BenchmarkResult("SQLAlchemy")
        
        # Setup
        engine = create_engine("sqlite:///sqlalchemy_benchmark.db")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Benchmark Create
        create_times = []
        for i in range(self.iterations):
            start_time = time.time()
            user = SQLAlchemyUser(
                username=f"user_{i}",
                email=f"user_{i}@example.com",
                is_active=True
            )
            session.add(user)
            session.commit()
            create_times.append(time.time() - start_time)
        
        result.add_result("Create", create_times)
        
        # Benchmark Read
        read_times = []
        for i in range(100):
            start_time = time.time()
            users = session.query(SQLAlchemyUser).all()
            read_times.append(time.time() - start_time)
        
        result.add_result("Read", read_times)
        
        # Benchmark Update
        update_times = []
        for i in range(100):
            start_time = time.time()
            session.query(SQLAlchemyUser).filter_by(username=f"user_{i}").update({"is_active": False})
            session.commit()
            update_times.append(time.time() - start_time)
        
        result.add_result("Update", update_times)
        
        # Benchmark Delete
        delete_times = []
        for i in range(100):
            start_time = time.time()
            session.query(SQLAlchemyUser).filter_by(username=f"user_{i}").delete()
            session.commit()
            delete_times.append(time.time() - start_time)
        
        result.add_result("Delete", delete_times)
        
        session.close()
        engine.dispose()
        return result
    
    def benchmark_peewee(self) -> BenchmarkResult:
        """Benchmark Peewee ORM - رفع مشکل و بهبود عملکرد"""
        if not PEWEE_AVAILABLE:
            print("Peewee not available, skipping benchmark")
            return None
            
        result = BenchmarkResult("Peewee")
        
        # Setup
        peewee_db.connect()
        peewee_db.create_tables([PeeweeUser, PeeweePost], safe=True)
        
        # Benchmark Create - بهبود عملکرد با استفاده از bulk insert
        create_times = []
        # ابتدا داده‌ها را جمع‌آوری می‌کنیم
        users_data = []
        for i in range(self.iterations):
            users_data.append({
                'username': f"user_{i}",
                'email': f"user_{i}@example.com",
                'is_active': True,
                'created_at': datetime.now()
            })
        
        # سپس به صورت گروهی اضافه می‌کنیم برای بهتر شدن عملکرد
        start_time = time.time()
        with peewee_db.atomic():
            for i in range(0, len(users_data), 100):  # batch size of 100
                batch = users_data[i:i+100]
                PeeweeUser.insert_many(batch).execute()
        create_times.append(time.time() - start_time)
        result.add_result("Create", [create_times[0] / self.iterations] * self.iterations)  # تقریب
        
        # Benchmark Read
        read_times = []
        for i in range(100):
            start_time = time.time()
            users = list(PeeweeUser.select())
            read_times.append(time.time() - start_time)
        
        result.add_result("Read", read_times)
        
        # Benchmark Update
        update_times = []
        for i in range(100):
            start_time = time.time()
            PeeweeUser.update(is_active=False).where(PeeweeUser.username == f"user_{i}").execute()
            update_times.append(time.time() - start_time)
        
        result.add_result("Update", update_times)
        
        # Benchmark Delete
        delete_times = []
        for i in range(100):
            start_time = time.time()
            PeeweeUser.delete().where(PeeweeUser.username == f"user_{i}").execute()
            delete_times.append(time.time() - start_time)
        
        result.add_result("Delete", delete_times)
        
        peewee_db.close()
        return result
	
    async def benchmark_tortoise(self) -> BenchmarkResult:
        """Benchmark Tortoise ORM"""
        if not TORTOISE_AVAILABLE:
            print("Tortoise not available, skipping benchmark")
            return None
        
        from tortoise import Tortoise
        
        result = BenchmarkResult("Tortoise")
        
        await Tortoise.init(
            db_url="sqlite://tortoise_benchmark.db",
            modules={"models": ["__main__"]}  # مطمئن شو مدل‌ها در __main__ هستن
        )
        await Tortoise.generate_schemas()
        
        # Create
        create_times = []
        for i in range(self.iterations):
            start_time = time.time()
            await TortoiseUser.create(
                username=f"user_{i}",
                email=f"user_{i}@example.com",
                is_active=True
            )
            create_times.append(time.time() - start_time)
        result.add_result("Create", create_times)

        # Read
        read_times = []
        for i in range(100):
            start_time = time.time()
            users = await TortoiseUser.all()
            read_times.append(time.time() - start_time)
        result.add_result("Read", read_times)

        # Update
        update_times = []
        for i in range(100):
            start_time = time.time()
            await TortoiseUser.filter(username=f"user_{i}").update(is_active=False)
            update_times.append(time.time() - start_time)
        result.add_result("Update", update_times)

        # Delete
        delete_times = []
        for i in range(100):
            start_time = time.time()
            await TortoiseUser.filter(username=f"user_{i}").delete()
            delete_times.append(time.time() - start_time)
        result.add_result("Delete", delete_times)

        await Tortoise.close_connections()
        return result

    async def run_all_benchmarks(self):
        """Run all available benchmarks"""
        print("🚀 Starting ORM Benchmark Suite")
        print(f"Testing with {self.iterations} iterations per operation")
        print("=" * 60)
        
        # Run Ormax benchmark
        if ORMAX_AVAILABLE:
            try:
                print("\nTesting Ormax...")
                ormax_result = await self.benchmark_ormax()
                if ormax_result:
                    self.results.append(ormax_result)
                    print(ormax_result)
            except Exception as e:
                print(f"Ormax benchmark failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Run SQLAlchemy benchmark
        if SQLALCHEMY_AVAILABLE:
            try:
                print("\nTesting SQLAlchemy...")
                sqlalchemy_result = self.benchmark_sqlalchemy_sync()
                if sqlalchemy_result:
                    self.results.append(sqlalchemy_result)
                    print(sqlalchemy_result)
            except Exception as e:
                print(f"SQLAlchemy benchmark failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Run Peewee benchmark
        if PEWEE_AVAILABLE:
            try:
                print("\nTesting Peewee...")
                peewee_result = self.benchmark_peewee()
                if peewee_result:
                    self.results.append(peewee_result)
                    print(peewee_result)
            except Exception as e:
                print(f"Peewee benchmark failed: {e}")
                import traceback
                traceback.print_exc()

        # Run Tortoise benchmark
        if TORTOISE_AVAILABLE:
            try:
                print("\nTesting Tortoise ORM...")
                tortoise_result = await self.benchmark_tortoise()
                if tortoise_result:
                    self.results.append(tortoise_result)
                    print(tortoise_result)
            except Exception as e:
                print(f"Tortoise benchmark failed: {e}")
                import traceback
                traceback.print_exc()

        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Sort results by total time
        sorted_results = sorted(
            [r for r in self.results if r], 
            key=lambda x: sum([op['total'] for op in x.results.values()])
        )
        
        if sorted_results:
            print(f"{'ORM':<15} {'Total Time':<12} {'Avg Time':<12} {'Rank'}")
            print("-" * 50)
            
            for i, result in enumerate(sorted_results):
                total_time = sum([op['total'] for op in result.results.values()])
                avg_time = statistics.mean([op['average'] for op in result.results.values()]) if result.results else 0
                rank = i + 1
                print(f"{result.orm_name:<15} {total_time:<12.4f} {avg_time:<12.6f} #{rank}")
            
            print(f"\n🏆 Winner: {sorted_results[0].orm_name}")
        else:
            print("No benchmarks completed successfully")


async def main():
    """Main benchmark function"""
    # Clean up any existing benchmark databases
    for db_file in ['ormax_benchmark.db',
'sqlalchemy_benchmark.db',
'peewee_benchmark.db',
'tortoise_benchmark.db']:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass
    
    # Run benchmarks
    benchmark = ORMBenchmark(iterations=1000)
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
