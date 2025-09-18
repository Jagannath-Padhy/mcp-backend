#!/usr/bin/env python3
"""
Complete Search to Cart Flow Test
Tests the full workflow from search to cart using MCP tools with real backend
"""

import asyncio
import json
import sys
from typing import Dict, Any

# Add to path for imports
sys.path.append('src')

# Import tools for testing
try:
    from src.mcp_server_fastmcp import (
        search_products, add_to_cart, view_cart, update_cart_quantity, 
        remove_from_cart, get_cart_total, clear_cart
    )
    from mcp.server.fastmcp import Context
    print("✅ All MCP tools imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class MockContext:
    """Mock context for testing tools"""
    def __init__(self):
        self.session_id = "search_cart_test_session"

async def test_complete_search_to_cart_flow():
    """Test complete search to cart workflow with backend compliance"""
    print("\n🔄 Complete Search to Cart Flow Test")
    print("=" * 60)
    
    ctx = MockContext()
    test_user_id = "mcpTestUser123"
    test_device_id = "mcpTestDevice456"
    
    try:
        # Step 1: Search for products
        print("📍 Step 1: Search for Apple Jam products...")
        search_result = await search_products(
            ctx,
            query="apple jam",
            max_results=5,
            userId=test_user_id,
            deviceId=test_device_id
        )
        print(f"   ✅ Search completed")
        
        # Parse search result to extract products
        search_data = json.loads(search_result)
        if not search_data.get('success'):
            print(f"   ❌ Search failed: {search_data.get('message')}")
            return False
            
        products = search_data.get('products', [])
        if not products:
            print("   ❌ No products found in search")
            return False
            
        print(f"   📦 Found {len(products)} products")
        first_product = products[0]
        print(f"   🎯 Selected: {first_product.get('name')} - ₹{first_product.get('price')}")
        
        # Step 2: Add product to cart
        print("\n📍 Step 2: Add product to cart...")
        add_result = await add_to_cart(
            ctx,
            item=first_product,
            quantity=2,
            userId=test_user_id,
            deviceId=test_device_id
        )
        
        add_data = json.loads(add_result)
        if not add_data.get('success'):
            print(f"   ❌ Add to cart failed: {add_data.get('message')}")
            return False
            
        print(f"   ✅ Added to cart successfully")
        print(f"   🛒 {add_data.get('message')}")
        
        # Step 3: View cart to verify
        print("\n📍 Step 3: View cart contents...")
        view_result = await view_cart(
            ctx,
            userId=test_user_id,
            deviceId=test_device_id
        )
        
        view_data = json.loads(view_result)
        if not view_data.get('success'):
            print(f"   ❌ View cart failed: {view_data.get('message')}")
            return False
            
        print(f"   ✅ Cart viewed successfully")
        print(f"   📝 {view_data.get('message')}")
        
        # Step 4: Get cart total
        print("\n📍 Step 4: Calculate cart total...")
        total_result = await get_cart_total(
            ctx,
            userId=test_user_id,
            deviceId=test_device_id
        )
        
        total_data = json.loads(total_result)
        if not total_data.get('success'):
            print(f"   ❌ Get cart total failed: {total_data.get('message')}")
            return False
            
        print(f"   ✅ Cart total calculated")
        print(f"   💰 {total_data.get('message')}")
        
        # Step 5: Update quantity (optional test)
        print("\n📍 Step 5: Update cart quantity (2 → 3)...")
        # We need to get cart items to find item ID for update
        cart_items = view_data.get('cart_items', [])
        if cart_items:
            item_id = cart_items[0].get('id') or cart_items[0].get('_id')
            if item_id:
                update_result = await update_cart_quantity(
                    ctx,
                    item_id=item_id,
                    quantity=3,
                    userId=test_user_id,
                    deviceId=test_device_id
                )
                
                update_data = json.loads(update_result)
                if update_data.get('success'):
                    print(f"   ✅ Quantity updated successfully")
                    print(f"   🔄 {update_data.get('message')}")
                else:
                    print(f"   ⚠️ Quantity update failed: {update_data.get('message')}")
            else:
                print("   ⚠️ Could not find item ID for quantity update")
        
        # Step 6: Clear cart (cleanup)
        print("\n📍 Step 6: Clear cart (cleanup)...")
        clear_result = await clear_cart(
            ctx,
            userId=test_user_id,
            deviceId=test_device_id
        )
        
        clear_data = json.loads(clear_result)
        if clear_data.get('success'):
            print(f"   ✅ Cart cleared successfully")
            print(f"   🧹 {clear_data.get('message')}")
        else:
            print(f"   ⚠️ Cart clear failed: {clear_data.get('message')}")
        
        print("\n🎉 Complete Search to Cart Flow: SUCCESS!")
        return True
        
    except Exception as e:
        print(f"\n❌ Flow test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_session_continuity():
    """Test that session data persists across multiple tool calls"""
    print("\n👤 Session Continuity Test")
    print("=" * 40)
    
    ctx = MockContext()
    test_user_id = "sessionTestUser"
    test_device_id = "sessionTestDevice"
    
    try:
        # Call 1: Search (should create session)
        print("📞 Call 1: Search products...")
        search_result = await search_products(
            ctx,
            query="apple",
            userId=test_user_id,
            deviceId=test_device_id
        )
        search_data = json.loads(search_result)
        session1 = search_data.get('session', {}).get('session_id')
        print(f"   Session ID: {session1}")
        
        # Call 2: Add to cart (should use same session)
        print("\n📞 Call 2: Add item to cart...")
        if search_data.get('products'):
            add_result = await add_to_cart(
                ctx,
                item=search_data['products'][0],
                quantity=1,
                userId=test_user_id,
                deviceId=test_device_id
            )
            add_data = json.loads(add_result)
            session2 = add_data.get('session', {}).get('session_id')
            print(f"   Session ID: {session2}")
            
            # Call 3: View cart (should use same session)
            print("\n📞 Call 3: View cart...")
            view_result = await view_cart(
                ctx,
                userId=test_user_id,
                deviceId=test_device_id
            )
            view_data = json.loads(view_result)
            session3 = view_data.get('session', {}).get('session_id')
            print(f"   Session ID: {session3}")
            
            # Verify session continuity
            if session1 == session2 == session3:
                print("\n✅ Session continuity: PASS")
                print(f"   All calls used same session: {session1}")
                return True
            else:
                print("\n❌ Session continuity: FAIL")
                print(f"   Sessions: {session1} → {session2} → {session3}")
                return False
        else:
            print("   ❌ No products found for session test")
            return False
            
    except Exception as e:
        print(f"\n❌ Session test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 ONDC Shopping MCP - Search to Cart Flow Test Suite")
    print("=" * 60)
    print("Testing backend compliance and session continuity...")
    
    # Test 1: Complete search to cart flow
    flow_success = await test_complete_search_to_cart_flow()
    
    # Test 2: Session continuity
    session_success = await test_session_continuity()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Complete Flow Test: {'PASS' if flow_success else 'FAIL'}")
    print(f"✅ Session Continuity: {'PASS' if session_success else 'FAIL'}")
    
    if flow_success and session_success:
        print("\n🎯 ALL TESTS PASSED! MCP tools are backend compliant! 🚀")
        return True
    else:
        print("\n⚠️ Some tests failed. Check logs above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)