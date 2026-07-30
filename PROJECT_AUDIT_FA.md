# گزارش بهینه‌سازی پروژه Ormax

## نتیجه نهایی

نسخه پروژه از `1.3.0` به `1.4.0` ارتقا داده شد. در محیط فعلی، مجموعه تست‌ها با نتیجه **48 passed** اجرا شد و کامپایل تمام فایل‌های پایتون نیز بدون خطا انجام گرفت.

## ایرادهای مهم رفع‌شده

### تراکنش‌ها و هم‌زمانی

- در بک‌اندهای دارای pool، دستورات `BEGIN`، کوئری‌ها و `COMMIT/ROLLBACK` ممکن بود روی اتصال‌های متفاوت اجرا شوند. اتصال تراکنش اکنون تا پایان context ثابت می‌ماند.
- وضعیت تراکنش از حالت مشترک خارج و به task جاری محدود شد.
- استفاده ناخواسته از context تراکنش در child task با خطای صریح متوقف می‌شود.
- تراکنش‌های هم‌زمان SQLite روی اتصال واحد سریال‌سازی شدند.
- در خطاها و cancellation، rollback اجرا می‌شود و خطای اصلی حفظ می‌گردد.
- تشخیص `microsoft://`، Aurora و Oracle در منطق تراکنش و savepoint اصلاح شد.

### SQLite

- وابستگی اجباری و اعلام‌نشده به `aiosqlite` حذف شد.
- fallback داخلی مبتنی بر `sqlite3` و `asyncio.to_thread` اضافه شد.
- commit خودکار هنگام اجرای فرمان‌های تراکنشی متوقف شد.
- مقدار `lastrowid` و `rowcount` به‌شکل صحیح‌تر بازگردانده می‌شود.

### مدل‌ها و فیلدها

- callable default برای هر instance جداگانه اجرا می‌شود.
- defaultهای mutable بین instanceها به اشتراک گذاشته نمی‌شوند.
- ارث‌بری فیلدها از مدل‌های abstract فعال شد.
- `class Meta` و `_meta` هر دو پشتیبانی می‌شوند.
- فیلد ناشناخته در سازنده مدل به‌جای نادیده گرفته‌شدن، خطا می‌دهد.
- تبدیل داده بین Python و Database در insert، update، filter و hydrate هماهنگ شد.
- `null=` به‌عنوان alias برای `nullable=` اضافه شد.
- رفتار Boolean، UUID، JSON و ForeignKey اصلاح شد.
- exceptionهای عمومی یکپارچه شدند.

### QuerySet و روابط

- `select_related()` برای روابط forward واقعاً eager loading انجام می‌دهد.
- `prefetch_related()` برای reverse relation لیست قابل‌استفاده و API سازگار با `all()` تولید می‌کند.
- `only()`، `defer()`، `values()` و `values_list()` اصلاح شدند و queryset اصلی را تغییر نمی‌دهند.
- `get()` در حالت چند نتیجه `MultipleObjectsReturned` می‌دهد.
- `count()` و `exists()` کوئری سبک‌تر اجرا می‌کنند.
- `limit`، `offset`، `timeout` و نام index اعتبارسنجی می‌شوند.
- فیلتر ForeignKey با خود instance مدل پشتیبانی می‌شود.
- `Model.get_related()` و `Model.set_related()` اضافه شدند.

### سازگاری بک‌اندها

- مسیر `mariadb://` به adapter صحیح MySQL/MariaDB متصل شد.
- وابستگی غیر استاندارد Oracle با driver رسمی `oracledb` و API async pool جایگزین شد.
- placeholderهای Oracle به bindهای `:1`، `:2` و ... تبدیل می‌شوند.
- ساخت و حذف جدول برای MSSQL و Oracle با syntax سازگار همان backend تولید می‌شود.

### ایندکس‌ها

- گزینه `index=True` قبلاً فقط ذخیره می‌شد و ایندکسی ایجاد نمی‌کرد؛ اکنون `create_tables()` ایندکس واقعی می‌سازد.
- `using_index()` از حالت بدون اثر خارج شد و برای SQLite، MySQL/MariaDB/Aurora و MSSQL hint تولید می‌کند.

## مستندات و بسته‌بندی

- README با API واقعی هماهنگ شد.
- benchmarkهای بدون منبع، ادعاهای غیرقابل‌اثبات و مثال‌های قابلیت‌های پیاده‌سازی‌نشده حذف شدند.
- محدودیت‌های فعلی پروژه شفاف ثبت شدند.
- `CHANGELOG.md` و `CONTRIBUTING.md` اضافه شدند.
- مستندات GitHub Pages بازسازی و workflow آن فقط به پوشه `docs` محدود شد.
- پشتیبانی بسته‌بندی روی Python 3.9 تا 3.13 تنظیم شد.
- marker نوع‌دهی `py.typed` اضافه شد.
- workflow تست چندنسخه‌ای برای GitHub Actions اضافه شد.

## اعتبارسنجی انجام‌شده

- `python -m compileall -q ormax tests`
- `python -m pytest -q`
- نتیجه: `48 passed`
- بررسی سازگاری grammar با Python 3.9
- بررسی metadata و ساخت source/wheel package
- نصب wheel در محیط مجزا و اجرای smoke test برای SQLite fallback، CRUD، rollback و index

## مواردی که نیاز به تست محیط واقعی دارند

در محیط تحویل، سرور یا driver واقعی PostgreSQL، MySQL/MariaDB/Aurora، MSSQL و Oracle در دسترس نبود. منطق اتصال و تراکنش آن‌ها اصلاح و از نظر ساختاری بررسی شده است، اما پیش از انتشار production باید integration test هر backend روی سرویس واقعی اجرا شود.

همچنین پروژه هنوز migration engine ندارد و متدهای `union()`، `intersection()` و `difference()` عمداً `NotImplementedError` می‌دهند. این موارد در مستندات جدید صریحاً ذکر شده‌اند.
